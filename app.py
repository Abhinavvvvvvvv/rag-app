import os
import streamlit as st
from ingest import ingest_documents
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.title("📄 PDF Question Answering (RAG)")

if not os.path.exists("faiss_index/index.faiss"):
    st.write("🔄 Setting up document intelligence...")
    ingest_documents()
    st.success("✅ Ready")

embeddings = OpenAIEmbeddings()
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = st.text_input("Ask a question")

if question:
    docs = db.similarity_search(question, k=3)
    for doc in docs:
        st.write(doc.page_content)

