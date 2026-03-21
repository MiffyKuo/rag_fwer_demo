class RiskControlledRAG:    # 拿 calibration 選好的最佳參數，真的去回答新問題
    def __init__(self, retriever, reranker, generator, best_params):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.best_params = best_params

    def answer(self, question: str):
        top_k = self.best_params["top_k"]
        top_K = self.best_params["top_K"]
        N_rag = self.best_params["N_rag"]

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        reranked = self.reranker.rerank(question, retrieved, top_K=top_K)
        contexts = reranked[:N_rag]
        answer = self.generator.generate_answer(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "top_k": top_k,
            "top_K": top_K,
            "N_rag": N_rag,
            "contexts": [d.page_content for d in contexts]
        }