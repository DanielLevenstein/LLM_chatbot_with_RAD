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
def get_instructions():
    return "Answer the following questions using simple straight forward language. "

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

def generate_llama_response(instruction, question, llm):
    # Combine user_prompt and system_message to create the prompt
    prompt = f"""Instructions: \n{instruction}\nQuestion: {question}\nAnswer:\n"""

    # Generate a response from the LLaMA model
    response = llm(
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=5,
        stop=['Instructions:', 'Question:', 'Answer:'],
        echo=False,
        seed=42,
    )
    # Extract the sentiment from the response
    response_text = response
    return response_text

if __name__ == '__main__':
    llm = create_llm(MODEL_PATH, MODEL_FILENAME)
    instructions = "You are a medical assistant chatbot who's job is to help doctors and nurses diagnose medical conditions. :"
    question1 = "What are the symptoms of Strep Throat. "
    response = generate_llama_response(
        instructions, question1, llm)
    response_text = response["choices"][0]["text"]
    print(response_text)
