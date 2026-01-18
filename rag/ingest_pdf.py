import pickle
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from llm_client import chunk_text, SENTENCE_TRANSFORMER
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

PDF_PATH = "data/blood_pressure_pdf.pdf"

OUTPUT_INDEX_PATH = "output/index.faiss"
OUTPUT_CHUNKS_PATH = "output/chunks.pkl"


if __name__ == '__main__':
    reader = PdfReader(PDF_PATH)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
    chunks = chunk_text(raw_text)

    model = SentenceTransformer(SENTENCE_TRANSFORMER)
    embeddings = model.encode(chunks, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "output/index.faiss")

    with open("output/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Ingested {len(chunks)} chunks into FAISS index")