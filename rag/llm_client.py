from langchain_core.messages import SystemMessage, HumanMessage
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_PATH = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

_client = None
def get_llm_client():
    global _client
    if _client is None:
        _client = create_llm(MODEL_PATH, MODEL_FILENAME)
    return _client

def create_llm(model_name_or_path, model_basename):
    # Using hf_hub_download to download a model from the Hugging Face model hub
    # The repo_id parameter specifies the model name or path in the Hugging Face repository
    # The filename parameter specifies the name of the file to download
    model_path = hf_hub_download(
        repo_id=model_name_or_path,
        filename=model_basename
    )
    llm = Llama(
        model_path=model_path,
        n_threads=2,  # CPU cores
        n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=43,  # Change this value based on your model and your GPU VRAM pool.
        n_ctx=4096,  # Context window
    )
    return llm

# Working.
def generate_response_without_context(llm, instruction: str, question: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        repeat_penalty=1.2,
    )

    return response["choices"][0]["message"]["content"].strip()

def generate_response_with_context(llm, instruction: str, context: str, question: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "system",
                "content": f"Retrieved context:\n{context}",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
        repeat_penalty=1.2,
    )
    return response["choices"][0]["message"]["content"].strip()