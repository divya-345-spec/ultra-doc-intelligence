import logging
import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.parser import parse_documents
from core.chunker import chunk_text
from core.retriever import build_vector_store, retrieve
from core.vector_store import VectorStore
from core.confidence import calculate_confidence
from core.guardrails import final_guardrail
from core.structured_extractor import extract_structured_fields
from core.puter_llm import puter_extract_structured_fields, puter_generate_answer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STRUCTURED_CACHE = {}

app = FastAPI(title="Ultra Doc Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    docs = parse_documents(UPLOAD_DIR)

    all_text = ""
    all_chunks = []
    for pages in docs.values():
        for page in pages:
            all_text += page["text"] + "\n"
        all_chunks.extend(chunk_text(pages))

    if all_chunks:
        store = build_vector_store(all_chunks)
        store.save()

    # Regex extraction (baseline)
    structured_data = extract_structured_fields(all_text)

    # LLM extraction (enhancement — fills missing fields)
    try:
        llm_data = await puter_extract_structured_fields(all_text)
        if llm_data:
            for key, value in llm_data.items():
                if value is not None:
                    structured_data[key] = value
    except Exception as e:
        logger.warning("LLM extraction failed, using regex only: %s", e)

    STRUCTURED_CACHE["data"] = structured_data

    return {
        "message": "File uploaded and processed successfully",
        "structured_fields": structured_data,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: AskRequest):
    question_lower = request.question.lower()

    # Check structured cache for direct field lookups
    structured_data = STRUCTURED_CACHE.get("data", {})

    FIELD_KEYWORDS = {
        "shipment_id": ["shipment id", "shipment number", "load id"],
        "shipper": ["shipper"],
        "consignee": ["consignee", "receiver"],
        "pickup_datetime": ["pickup date", "pickup time", "pick up"],
        "delivery_datetime": ["delivery date", "delivery time", "deliver"],
        "equipment_type": ["equipment type", "equipment"],
        "mode": ["mode of transport", "shipping mode"],
        "rate": ["carrier rate", "rate", "charge", "amount"],
        "currency": ["currency"],
        "weight": ["weight"],
        "carrier_name": ["carrier name", "carrier"],
    }

    for field, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw in question_lower:
                value = structured_data.get(field)
                if value:
                    return {
                        "answer": value,
                        "confidence": 0.95,
                        "sources": [{"page": "structured", "text": f"Extracted from {field}"}],
                    }
                break

    # RAG retrieval
    store = VectorStore.load()

    if store is None:
        docs = parse_documents(UPLOAD_DIR)
        all_chunks = []
        for pages in docs.values():
            all_chunks.extend(chunk_text(pages))

        if not all_chunks:
            return {"answer": "No documents uploaded yet.", "confidence": 0.0, "sources": []}

        store = build_vector_store(all_chunks)

    results = retrieve(request.question, store)

    safe = final_guardrail(request.question, results)

    if not safe or not results:
        return {"answer": "Not found in document.", "confidence": 0.0, "sources": []}

    confidence = calculate_confidence(results)
    top_chunks = [r["chunk"] for r in results[:3]]

    # LLM answer generation
    llm_answer = None
    try:
        llm_answer = await puter_generate_answer(request.question, top_chunks)
    except Exception as e:
        logger.warning("LLM answer generation failed: %s", e)

    if llm_answer and "not found in document" not in llm_answer.lower():
        answer = llm_answer
    else:
        answer = results[0]["chunk"]["text"][:500]

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": [
            {"page": r["chunk"]["page"], "text": r["chunk"]["text"][:200]}
            for r in results[:3]
        ],
    }


@app.post("/extract")
async def extract_data():
    docs = parse_documents(UPLOAD_DIR)

    all_text = ""
    for pages in docs.values():
        for page in pages:
            all_text += page["text"] + "\n"

    empty_result = {
        "shipment_id": None, "shipper": None, "consignee": None,
        "pickup_datetime": None, "delivery_datetime": None,
        "equipment_type": None, "mode": None, "rate": None,
        "currency": None, "weight": None, "carrier_name": None,
    }

    if not all_text.strip():
        return empty_result

    extracted = extract_structured_fields(all_text)

    # LLM enhancement
    try:
        llm_data = await puter_extract_structured_fields(all_text)
        if llm_data:
            for key, value in llm_data.items():
                if value is not None:
                    extracted[key] = value
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)

    # Ensure all 11 required keys
    for key in empty_result:
        if key not in extracted:
            extracted[key] = None

    return extracted


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
