# Ultra Doc Intelligence

AI-powered logistics document analysis platform. Upload shipping documents (PDF, DOCX, TXT), ask natural language questions, and extract structured shipment data — all powered by RAG with hallucination guardrails.

## Features

- **Document Upload** — Supports PDF, DOCX, and TXT files
- **RAG-based Q&A** — Ask questions about uploaded documents with confidence scores and source references
- **Structured Extraction** — Extracts 11 logistics fields as JSON:
    
    `shipment_id`, `shipper`, `consignee`, `pickup_datetime`, `delivery_datetime`, `equipment_type`, `mode`, `rate`, `currency`, `weight`, `carrier_name`
    
- **Hallucination Guardrails** — Blocks answers when retrieval confidence is low or content doesn't match the query
- **Optional LLM Enhancement** — Supports REST-based LLM integration to improve extraction and generate natural language answers; automatically falls back to regex and retrieval-based responses if LLM is unavailable

## Tech Stack

| Component | Technology |
| --- | --- |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Vector Store | FAISS (IndexFlatL2), persisted to disk |
| Embeddings | Character-level hashing (128-dim, deterministic, no API needed) |
| LLM Integration | Optional REST-based LLM (fallback supported) |
| Document Parsing | pypdf, python-docx |

## Project Structure

```
ultradoc/
├── app/
│   ├── main.py# FastAPI app — /upload, /ask, /extract endpoints
│   ├── core/
│   │   ├── parser.py# PDF / DOCX / TXT parsing
│   │   ├── chunker.py# Text chunking with overlap
│   │   ├── retriever.py# Embedding + FAISS retrieval
│   │   ├── vector_store.py# FAISS index with save/load persistence
│   │   ├── confidence.py# Distance-to-confidence scoring
│   │   ├── guardrails.py# Hallucination prevention guardrails
│   │   ├── structured_extractor.py# Regex-based field extraction (11 fields)
│   │   └── puter_llm.py# Optional REST API integration for LLM calls
│   └── data/
│       ├── uploads/# Uploaded documents stored here
│       └── vector_store/# Persisted FAISS index + chunks
├── ui/
│   └── app.py# Streamlit frontend
├── .env# Optional LLM credentials (not committed)
└── requirement.txt# Python dependencies
```

## Setup

### 1. Clone and enter the project

```bash
cd ultradoc
```

### 2. Create a conda environment (or use an existing one)

```bash
conda create -n ultradoc python=3.11 -y
conda activate ultradoc
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. (Optional) Configure environment variables

Create a `.env` file in the project root **only if enabling LLM enhancement**:

```
PUTER_USERNAME=your_username
PUTER_PASSWORD=your_password
PUTER_MODEL=gpt-4o
```

> LLM integration is optional.
> 
> 
> If credentials are not provided or authentication fails, the application automatically falls back to regex-based extraction and retrieval-grounded answers without breaking functionality.
> 

## Running the App

You need **two terminals** — one for the API, one for the UI.

### Terminal 1 — Start the FastAPI backend

```bash
cd app
python main.py
```

The API starts at `http://127.0.0.1:8000`.

### Terminal 2 — Start the Streamlit frontend

```bash
streamlit run ui/app.py
```

The UI opens at `http://localhost:8501`.

## API Endpoints

### `GET /`

Health check.

```json
{"status":"API is running"}
```

### `POST /upload`

Upload a document for processing.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` (PDF, DOCX, or TXT)

```bash
curl -F"file=@document.pdf" http://127.0.0.1:8000/upload
```

**Response:**

```json
{"message":"File uploaded and processed successfully","structured_fields":{"shipment_id":"SH-12345","shipper":"ABC Logistics","rate":"2500","currency":"USD",
    ...}}
```

### `POST /ask`

Ask a natural language question about uploaded documents.

- **Content-Type:** `application/json`
- **Body:** `{ "question": "What is the carrier rate?" }`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H"Content-Type: application/json" \
  -d'{"question": "What is the carrier rate?"}'
```

**Response:**

```json
{"answer":"The carrier rate is 2500 USD.","confidence":0.81,"sources":[{"page":1,"text":"Carrier Rate: 2500 USD ..."}]}
```

### `POST /extract`

Extract all 11 structured fields from uploaded documents.

```bash
curl -X POST http://127.0.0.1:8000/extract
```

**Response:**

```json
{"shipment_id":"SH-12345","shipper":"ABC Logistics","consignee":"XYZ Corp","pickup_datetime":"2024-01-15","delivery_datetime":"2024-01-18","equipment_type":"Dry Van","mode":"FTL","rate":"2500","currency":"USD","weight":"15000 lbs","carrier_name":"Fast Freight Inc"}
```

## How It Works

1. **Upload** — Document is parsed page-by-page, chunked (500 chars, 100 overlap), embedded using character-level hashing, and indexed in FAISS. Regex extraction runs by default; optional LLM enhancement improves results when configured.
2. **Ask** — Question is first checked against cached structured fields for direct matches. If no match, RAG retrieves top chunks from FAISS. Guardrails verify the answer is grounded in the document. The system returns grounded text; optional LLM enhancement refines formatting when enabled.
3. **Extract** — Re-parses documents, runs regex extraction for all 11 fields, then optionally enhances with LLM. Returns JSON with `null` for any field not found.

## Guardrails

The system includes four guardrails to prevent hallucination:

| Guardrail | Purpose |
| --- | --- |
| Strong Retrieval | Blocks if best FAISS distance > 1.5 (no relevant chunk) |
| Keyword Intent | Blocks if question asks for financial IDs (IFSC, SWIFT, IBAN) not present in text |
| Question Coverage | Blocks if fewer than 20% of content words appear in retrieved text |
| Rate Guardrail | Blocks rate/charge answers unless numeric + currency evidence exists in text |