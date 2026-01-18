import rag.llm_client
from rag import llm_client

SYSTEM_PROMPT = "Answer the following questions using simple straight forward language. "
SYSTEM_PROMPT2 = '''SYSTEM: If the answer is not contained in the provided context, respond with:
"I don’t know based on the given information, do not use information found outside of the system context."'''
SYSTEM_PROMPT3 = '''
You are a strict RAG assistant.
- Use ONLY the information provided in the retrieved context.
- If the context contains the answer, provide it concisely.
- If the context does NOT contain the answer, respond exactly:
  "I don’t know based on the given information."
- Do not use prior knowledge or guess.
'''

class ChatBot:
    def __init__(self):
        self.llm = rag.llm_client.get_llm_client()
    def ask_question_without_context(self, message: str) -> str:
        return llm_client.generate_response_without_context(self.llm, SYSTEM_PROMPT, message)
    def ask_question_with_context(self, context: str, message: str) -> str:
        return llm_client.generate_response_with_context(self.llm, SYSTEM_PROMPT3, context, message)

if __name__ == '__main__':
    chatbot = ChatBot()
    response = chatbot.ask_question_without_context("What model is this chatbot using?")
    print(response)
    response2 = chatbot.ask_question_with_context(f"Model is using {llm_client.MODEL_FILENAME}", "What model is this chatbot using?")
    print(response2)
    response3 = chatbot.ask_question_with_context(f"Model is using {llm_client.MODEL_FILENAME}", "What GPU is the model running on?")
    print(response3)
    response4 = chatbot.ask_question_with_context("The sky is blue", "What color is the sky?")
    print(response4)
    response5 = chatbot.ask_question_with_context("", "What color is the sky?")
    print(response5)