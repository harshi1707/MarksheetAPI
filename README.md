# MarksheetAPI: Marksheet Extraction API

This project is a high-performance, containerized FastAPI that extracts structured data from academic marksheets. It leverages a hybrid approach combining computer vision and LLMs to achieve high accuracy and confidence in data extraction. The API supports various formats (JPG, PNG, PDF) and is designed for scalability and robust error handling.

## Quickstart
1. Copy files to a folder.
2. Create and fill `.env` from `.env.example` with your OpenAI key.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run locally: `uvicorn app.main:app --reload`.
5. Open Swagger UI: http://localhost:8000/docs

## Endpoints
- `POST /extract` — single-file extract
- `POST /extract/batch` — (zip) batch processing

## Notes
- Keep `OPENAI_API_KEY` out of git. Use platform env vars when deploying.
- This project uses EasyOCR + OpenAI for normalization. You can swap the LLM client.