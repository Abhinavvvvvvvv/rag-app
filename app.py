import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="PDF Question Answering (RAG)")
st.title("📄 PDF Question Answering (RAG)")

st.write("Ask questions based on the uploaded PDF knowledge base.")

# Dummy embeddings object (won't actually call OpenAI)
embeddings = OpenAIEmbeddings()

# Load existing FAISS index (already built locally)
try:
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error("FAISS index not found. Please add the faiss_index folder.")
    st.stop()

query = st.text_input("Ask a question")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Answer from PDF:")
    st.write(docs[0].page_content)



