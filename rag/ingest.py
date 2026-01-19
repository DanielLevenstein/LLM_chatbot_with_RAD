import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from llm_client import chunk_text, SENTENCE_TRANSFORMER

DATA_DIR = "data"
OUTPUT_INDEX_PATH = "output/index.faiss"
OUTPUT_CHUNKS_PATH = "output/chunks.pkl"


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def ingest_folder(data_dir: str):
    all_chunks: list[str] = []

    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)

        if not os.path.isfile(path):
            continue

        print(f"Processing: {filename}")

        try:
            if filename.lower().endswith(".txt"):
                raw_text = read_txt(path)

            elif filename.lower().endswith(".pdf"):
                raw_text = read_pdf(path)

            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return all_chunks

def embed_chunks(chunks):
    model = SentenceTransformer(SENTENCE_TRANSFORMER)
    embeddings = model.encode(chunks, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "output/index.faiss")

    with open("output/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Ingested {len(chunks)} chunks into FAISS index")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # 1 Load and chunk all files
    chunks = ingest_folder(DATA_DIR)

    if not chunks:
        raise RuntimeError("No chunks were generated. Check your data folder.")

    print(f"Total chunks created: {len(chunks)}")
    embed_chunks(chunks)

