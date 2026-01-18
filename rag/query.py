import faiss
import pickle
import numpy as np
def load_chunks(path="output/chunks.pkl"):
    with open(path, "rb") as f:
        chunks = pickle.load(f)

    # sanity checks
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks, got {type(chunks)}")
    if len(chunks) == 0:
        raise ValueError("Loaded chunks list is empty")
    if not all(isinstance(c, str) for c in chunks):
        raise ValueError("All chunks must be strings")

    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # Example: inspect the first chunk
    print(type(chunks))  # should be <class 'list'>
    print(len(chunks))  # should match number of FAISS vectors
    print(chunks[0])  # should be a text chunk