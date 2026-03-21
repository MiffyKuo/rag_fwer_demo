from langchain_ollama import ChatOllama

class GeneratorModule:
    def __init__(self, model_name: str):
        self.llm = ChatOllama(model=model_name, temperature=0)

    def generate_answer(self, question: str, contexts: list):
        context_text = "\n\n".join(
            [f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(contexts)]
        )

        prompt = f"""
You are a QA assistant.
Answer the question only based on the provided context.
If the context is insufficient, say you are unsure.

Question:
{question}

Context:
{context_text}

Return only the answer text.
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)