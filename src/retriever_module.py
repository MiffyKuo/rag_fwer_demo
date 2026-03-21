from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RetrieverModule:
    def __init__(self, embedding_model: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    def build_index(self, corpus):  # 把文件變成向量資料庫
        docs = [
            Document(page_content=row["text"], metadata={"doc_id": row["doc_id"]})
            for row in corpus
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, question: str, top_k: int):  # 對問題找最相似的 top-k 文件
        return self.vectorstore.similarity_search(question, k=top_k)