# 拿 calibration 選好的最佳參數，真的去回答新問題
class RiskControlledRAG:
    def __init__(self, retriever, reranker, generator, best_params):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.best_params = best_params

    def answer(self, question: str):
        top_k = self.best_params["top_k"]
        top_K = self.best_params["top_K"]
        N_rag = self.best_params["N_rag"]
        lambda_g = self.best_params["lambda_g"]
        lambda_s = self.best_params["lambda_s"]

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        reranked = self.reranker.rerank(question, retrieved, top_K=top_K)
        contexts = reranked[:N_rag]

        answers = self.generator.generate_answers(
            question,
            contexts,
            lambda_g=lambda_g,
            lambda_s=lambda_s
        )

        final_answer = answers[0] if len(answers) > 0 else "I do not know."

        return {
            "question": question,
            "answer": final_answer,
            "candidate_answers": answers,
            "top_k": top_k,
            "top_K": top_K,
            "N_rag": N_rag,
            "lambda_g": lambda_g,
            "lambda_s": lambda_s,
            "contexts": [d.page_content for d in contexts],
        }