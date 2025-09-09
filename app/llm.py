import os
import json
import re
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")

# Initialize client
client = OpenAI(api_key=api_key)

# Default model
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def build_lm_prompt(blocks: list) -> str:
    sample_blocks = blocks[:400]
    blk_text = "\n".join(
        [f"- {b['text']} (ocr_conf={b['conf']:.2f})" for b in sample_blocks]
    )

    return f"""
You are a JSON-only marksheet parser.
Input: OCR blocks (text + ocr_confidence).

Task:
- Extract candidate fields (name, father_name, mother_name, dob, roll_no, registration_no, exam_year, board, institution).
- Extract subject rows (subject_name, max_marks, obtained_marks, grade).
- Extract overall totals and result.
- Extract issue date/place if present.

Output Rules:
- Each extracted field must include: value (string/number/null) and llm_confidence (0.0–1.0).
- Normalize dates to YYYY-MM-DD if possible.
- Normalize numbers where possible.
- Respond ONLY with valid JSON matching this structure:
{{
  "candidate": {{...}},
  "subjects": [{{...}}],
  "overall": {{...}},
  "issue": {{...}}
}}

OCR_BLOCKS:
{blk_text}

Confidence guidelines:
- If uncertain, use 0.3–0.6
- If exact match from OCR with clear numeric format, use >0.9
"""


def call_llm_parse(blocks: list) -> Dict[str, Any]:
    prompt = build_lm_prompt(blocks)

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful parser."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )

        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.M).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\})", text, re.S)
            if m:
                return json.loads(m.group(1))

        return {"error": "No valid JSON found in LLM response"}

    except Exception as e:
        return {"error": str(e)}
