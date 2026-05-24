# AWS Documentation RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) chatbot built for answering technical AWS questions using 
live documentation embeddings and semantic search.

The system crawls selected AWS documentation pages, processes and chunks the content, generates vector embeddings, 
and stores them in a FAISS index for fast retrieval. When a user submits a question, the chatbot retrieves the 
most relevant documentation fragments and uses them as grounded context for response generation.

## Current AWS Coverage

```
["cli", "cloudformation", "cloudwatch", "dynamodb", "elasticloadbalancing",
            "ec2", "ecs", "eks", "iam", "lambda", "rds", "s3",
            "sagemaker", "vpc", "xray" ]

```


## 1. Install Dependencies

Make sure your virtual environment is active and install required packages:

```bash
pip install -r requirements.txt
```

## Running Locally
My AWS rag includes a local streamlit app for testing. For ease of testing, the `extract.py` and `ingest.py` scripts 
have been run and the chunks and indexes created have been committed to source control.  

To run the streamlit app run, install the dependencies an run the following command locally. 

```streamlit run app.py```

## Example Queries

- "How do I configure an Application Load Balancer for ECS?"
- "What permissions are required for Lambda to access S3?"
- "How do I troubleshoot DynamoDB throttling?"
- "How do I deploy a SageMaker endpoint?"

# Architecture Overview

1. Documentation Crawling

   - Scrapes AWS documentation pages
   - Limits recursive traversal depth to control corpus size
   - Filters and normalizes extracted content
2. Document Processing

   - Cleans HTML and removes navigation noise
   - Splits documentation into searchable chunks
   - Preserves contextual structure where possible
3. Embedding & Indexing

   - Generates vector embeddings for document chunks
   - Stores embeddings in a FAISS vector index
   - Enables low-latency semantic similarity search
4. Retrieval-Augmented Generation

   - Retrieves relevant documentation chunks for each query
   - Injects retrieved context into the LLM prompt
   - Produces grounded technical responses based on AWS documentation

## Goals

- Reduce hallucinations in AWS-related answers
- Provide context-aware technical support
- Enable fast semantic search across AWS services
- Serve as a lightweight proof-of-concept RAG architecture

## Tech Stack

- Python
- BeautifulSoup
- Requests
- FAISS
- Embedding Models
- Large Language Models (LLMs)


## Status

This project is currently a proof of concept focused on validating:

- constrained documentation crawling
- semantic retrieval quality
- FAISS-based vector search
- AWS-focused RAG workflows
### Notes

- Make sure the embedding model used during ingestion matches the model used for querying (default: `all-MiniLM-L6-v2`).
- FAISS indexes and chunks must remain aligned — do not modify the pickled chunks file manually.
- All scripts have to be run from the root directory of the project.

# Creating New Indexes

## Ingest Data

You can ingest documents into the RAG pipeline the ingestion scripts: `rag/ingest.py`

ingest.py can handle txt or pdf documents

This process will:

1. Read your document(s)
2. Chunk the text into manageable pieces
3. Generate embeddings using the SentenceTransformer model
4. Build a FAISS index
5. Save the chunks and index in the `index/` folder

### Output

- `index/index.faiss` – FAISS vector index
- `index/chunks.pkl` – Pickled text chunks

