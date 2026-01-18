from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
# Model Options.
# Qwen3-4B-BF16.gguf, 8.05 GB
# Qwen3-4B-Q8_0.gguf, 4.28 GB, 53 s
# Qwen3-4B-Q6_K.gguf, 3.31 GB
# Qwen3-4B-Q5_K_M.gguf, 2.85 GB

MODEL_PATH = "unsloth/Qwen3-4B-GGUF"
MODEL_FILENAME = "Qwen3-4B-Q5_K_M.gguf"
_client = None
def get_llm_client():
    global _client
    if _client is None:
        _client = create_llm(MODEL_PATH, MODEL_FILENAME)
    return _client

def create_llm(model_name_or_path, model_basename):
    print(f"Creating model: {MODEL_FILENAME}")
    start_time = datetime.now()
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
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"{MODEL_FILENAME} creating time: {elapsed_time}")
    return llm

# Working.
def generate_response_without_context(llm, instruction: str, question: str) -> str:
    start_time = datetime.now()
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

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"generate_response_without_context -Elapsed time: {elapsed_time}")
    return trim_response(response["choices"][0]["message"]["content"])

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
    return trim_response(response["choices"][0]["message"]["content"])

def trim_response(response_text):
    marker = "</think>"
    response_text = response_text
    if marker in response_text:
        return response_text.split(marker, 1)[1].lstrip()
    return response_text
