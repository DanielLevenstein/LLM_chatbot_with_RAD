import rag_chatbot.llm_client
from rag_chatbot import llm_client
SYSTEM_PROMPT = "Answer the following questions using simple straight forward language. "

class ChatBot:
    def __init__(self):
        self.llm = rag_chatbot.llm_client.get_llm_client()
    def ask(self, message: str) -> str:
        return llm_client.generate_llama_response(self.llm, SYSTEM_PROMPT, "", message)

if __name__ == '__main__':
    chatbot = ChatBot()
    response = chatbot.ask("What model is this chatbot using?")
    print(response["choices"][0]["text"])
