# RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying documents using an LLM. It supports ingesting both **plain text** and **PDF files**.

## Getting Started

### 1. Install Dependencies

Make sure your virtual environment is active and install required packages:

```bash
pip install -r requirements.txt
```

Required packages include `faiss`, `sentence-transformers`, and `pymupdf` (for PDF ingestion).

## Ingest Data

You can ingest documents into the RAG pipeline the ingestion scripts: `rag/ingest.py`

ingest.py can handle txt or pdf documents

This process will:

1. Read your document(s)
2. Chunk the text into manageable pieces
3. Generate embeddings using the SentenceTransformer model
4. Build a FAISS index
5. Save the chunks and index in the `output/` folder

### Output

- `output/index.faiss` – FAISS vector index
- `output/chunks.pkl` – Pickled text chunks

## Local Testing

After ingestion run the streamlit app using the following command. 
`streamlit run app.py`

### Notes

- Make sure the embedding model used during ingestion matches the model used for querying (default: `all-MiniLM-L6-v2`).
- FAISS indexes and chunks must remain aligned — do not modify the pickled chunks file manually.
- All scripts have to be run from the root directory of the project.
- blood_pressure_PDF.pdf is a stand in file for a companies' internal datastore.
