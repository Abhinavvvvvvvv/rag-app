import faiss
import pickle
import numpy as np

def load_rag():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("faiss_index/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return index, chunks, vectorizer

def ask_question(question, index, chunks, vectorizer, k=3):
    q_vec = vectorizer.transform([question]).toarray()
    distances, indices = index.search(q_vec, k)

    answers = []
    for i in indices[0]:
        answers.append(chunks[i])

    return "\n---\n".join(answers)

if __name__ == "__main__":
    index, chunks, vectorizer = load_rag()

    print("Ask a question (type 'exit' to quit)")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        result = ask_question(q, index, chunks, vectorizer)
        print("\nAI (from PDF):\n", result)
        print("\n")
