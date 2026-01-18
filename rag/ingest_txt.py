import faiss
import pickle
from sentence_transformers import SentenceTransformer

from llm_client import chunk_text, SENTENCE_TRANSFORMER
OUTPUT_INDEX_PATH = "output/index.faiss"
OUTPUT_CHUNKS_PATH = "output/chunks.pkl"


if __name__ == '__main__':
    with open("data/blood_pressure_info.txt") as f :
        raw_text = f.read()
        chunks = chunk_text(raw_text)
        model = SentenceTransformer(SENTENCE_TRANSFORMER)
        embeddings = model.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, "output/index.faiss")

    with open("output/chunks.pkl", "wb") as f:
        # Ignore this error
        pickle.dump(chunks, f)

    print(f"Ingested({len(chunks)} chunks) into {len(chunks)} chunks")