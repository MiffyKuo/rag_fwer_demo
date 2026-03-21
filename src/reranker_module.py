from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleReranker: # without cross-encoder (先測試)
    def __init__(self, embedding_model: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    def rerank(self, question: str, docs: list, top_K: int):
        q_vec = np.array(self.embeddings.embed_query(question)).reshape(1, -1)
        scored = []

        for doc in docs:
            d_vec = np.array(self.embeddings.embed_query(doc.page_content)).reshape(1, -1)
            score = cosine_similarity(q_vec, d_vec)[0, 0]
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_K]]