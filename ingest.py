import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def create_vector_store(pdf_path):
    print("Reading PDF...")
    text = read_pdf(pdf_path)

    print("Splitting text...")
    chunks = chunk_text(text)
    print(f"Chunks created: {len(chunks)}")

    print("Creating TF-IDF embeddings...")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks).toarray()

    print("Saving FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs("faiss_index", exist_ok=True)

    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("faiss_index/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("DONE ✅ FAISS CREATED")

if __name__ == "__main__":
    create_vector_store("data/test.pdf")
