import os
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
# app.py
# Add these statements to import and initialise AppSignal
import appsignal
appsignal.start()

# Make sure you initialise AppSignal before importing Flask!
from flask import Flask # noqa: E402
app = Flask(__name__)

# your app code

# Importing modules
from app.ocr import  run_ocr
from app.llm import call_llm_parse
from app.utils import combine_confidences, best_ocr_conf_for_value

# Global thread pool for CPU-bound tasks (OCR, LLM)
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Marksheet Extraction API", version="1.0.0")

# Asynchronous wrapper for the blocking OCR function
async def run_ocr_async(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    return await run_in_threadpool(run_ocr, file_bytes, filename)

# Asynchronous wrapper for the blocking LLM function
async def call_llm_parse_async(ocr_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return await run_in_threadpool(call_llm_parse, ocr_blocks)

# Helper function to process and enrich a field with confidence scores
def enrich_field(value_dict: Optional[Dict[str, Any]], ocr_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(value_dict, dict) or "value" not in value_dict:
        return {"value": None, "confidence": 0.0, "bbox": None}
    
    val = value_dict.get("value")
    llm_c = float(value_dict.get("llm_confidence", 0.5))
    ocr_c, bbox = best_ocr_conf_for_value(str(val), ocr_blocks)

    return {
        "value": val,
        "confidence": combine_confidences(ocr_c, llm_c),
        "bbox": bbox,
        "meta": {"ocr": ocr_c, "llm": llm_c}
    }

@app.post("/parse", response_model=Dict[str, Any])
async def parse_marksheet(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Marksheet Extraction Pipeline:
    1. Accept file (JPG, JPEG, PNG, PDF).
    2. Run OCR to extract text blocks.
    3. Send blocks to LLM for structured parsing.
    4. Fuse OCR + LLM confidence scores.
    """
    filename = file.filename.lower()
    
    # Validation
    if not filename.endswith((".jpg", ".jpeg", ".png", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPG/PNG/PDF.")
        
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    # Step 1: OCR (runs in a separate thread)
    try:
        ocr_blocks = await run_ocr_async(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
        
    if not ocr_blocks:
        raise HTTPException(status_code=422, detail="OCR failed. No text detected.")

    # Step 2: LLM Parsing (runs in a separate thread)
    try:
        llm_output = await call_llm_parse_async(ocr_blocks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {e}")
        
    if "error" in llm_output:
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {llm_output['error']}")

    # Step 3: Fuse confidences and build the final response
    final_output = {
        "candidate": {k: enrich_field(v, ocr_blocks) for k, v in llm_output.get("candidate", {}).items()},
        "subjects": [
            {k: enrich_field(v, ocr_blocks) for k, v in subj.items()}
            for subj in llm_output.get("subjects", [])
        ],
        "overall": {k: enrich_field(v, ocr_blocks) for k, v in llm_output.get("overall", {}).items()},
        "issue": {k: enrich_field(v, ocr_blocks) for k, v in llm_output.get("issue", {}).items()}
    }

    return final_output

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)