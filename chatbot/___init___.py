from rag.llm_client import get_llm_client, load_rag_assets

class ChatBot:
    def __init__(self):
        self.llm = get_llm_client()
        self.index, self.chunks = load_rag_assets()