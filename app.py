import os
import streamlit as st
from ingest import ingest_documents
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 PDF Question Answering (RAG)")

if not os.path.exists("faiss_index"):
    st.info("Building knowledge base... please wait ⏳")
    ingest_documents()
    st.success("Knowledge base created!")

embeddings = OpenAIEmbeddings()
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = st.text_input("Ask a question")

if query:
    docs = db.similarity_search(query, k=3)
    st.write(docs[0].page_content)


