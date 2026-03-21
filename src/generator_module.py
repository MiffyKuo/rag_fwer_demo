from langchain_ollama import ChatOllama

class GeneratorModule:
    def __init__(self, model_name: str):
        self.llm = ChatOllama(model=model_name, temperature=0)

#     def generate_answer(self, question: str, contexts: list):
#         context_text = "\n\n".join(
#             [f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(contexts)]
#         )

#         prompt = f"""
# You are a QA assistant.
# Answer the question only based on the provided context.
# If the context is insufficient, say you are unsure.

# Question:
# {question}

# Context:
# {context_text}

# Return only the answer text.
# """
#         response = self.llm.invoke(prompt)
#         return response.content if hasattr(response, "content") else str(response)
    def generate_answers(self, question, contexts, lambda_g=3, lambda_s=0.8, max_retry=20):
        prompt = self.build_prompt(question, contexts)
        answers = []

        from difflib import SequenceMatcher

        def sim(a, b):
            return SequenceMatcher(None, a, b).ratio()

        tries = 0
        while len(answers) < lambda_g and tries < max_retry:
            response = self.llm.invoke(prompt)
            candidate = response.content.strip()

            if candidate and all(sim(candidate, old) <= lambda_s for old in answers):
                answers.append(candidate)

            tries += 1

        if len(answers) == 0:
            response = self.llm.invoke(prompt)
            answers.append(response.content.strip())

        return answers