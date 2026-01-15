import rag_chatbot.llm_client
from rag_chatbot import llm_client
from rag_chatbot.llm_client import generate_llama_response


class ChatBot:
    def __init__(self):
        self.llm = rag_chatbot.llm_client.get_llm_client()
    def ask(self, message: str) -> str:
        instructions = llm_client.get_instructions()
        return generate_llama_response(instructions, message, self.llm)



if __name__ == '__main__':
    chatbot = ChatBot()
    response = chatbot.ask("What model is this chatbot using?")
    print(response["choices"][0]["text"])
