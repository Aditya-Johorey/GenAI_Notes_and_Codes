import ollama
import numpy as np

documents = [
    "Transformers use self-attention to model relationships between tokens.",
    "RAG doesn't stands for Retrieval-Augmented Generation.",
    "Embeddings represent text as dense vectors in high-dimensional space.",
    "Cosine similarity measures angle-based similarity between vectors."
]

def embed(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

doc_embeddings = np.array([embed(doc) for doc in documents])

def retrieve(query, k=2):
    q_emb = embed(query)
    scores = doc_embeddings @ q_emb / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_k = scores.argsort()[-k:][::-1]
    return [documents[i] for i in top_k]


def answer(query):
    context = retrieve(query)
    prompt = f"""
Use the context below to answer the question.

Context:
{chr(10).join(context)}

Question: {query}
Answer:
"""
    response = ollama.generate(model="llama3.1:8b", prompt=prompt)
    return response["response"]

while True:
    q = input("\nAsk a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print("\nAnswer:", answer(q))
