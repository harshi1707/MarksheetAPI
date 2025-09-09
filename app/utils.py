from typing import Dict, Any, List


def combine_confidences(ocr_c: float, llm_c: float, w_ocr: float = 0.6, w_llm: float = 0.4) -> float:
    """
    Combine OCR and LLM confidence scores using a weighted average.
    Ensures output is clamped between 0.0 and 1.0.
    Default weights: OCR (0.6), LLM (0.4).
    """
    try:
        score = w_ocr * float(ocr_c) + w_llm * float(llm_c)
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, score))


def best_ocr_conf_for_value(value: str, ocr_blocks: List[Dict]) -> float:
    """
    Given a candidate value (string) and OCR blocks, return the best matching OCR confidence.
    Matching strategies:
    1. Exact match (case-insensitive).
    2. Substring containment (value in OCR text or vice versa).
    3. Fuzzy token overlap (gives partial credit).
    """
    if not value:
        return 0.0

    val = str(value).strip().lower()
    if not val:
        return 0.0

    best = 0.0
    for b in ocr_blocks:
        t = str(b.get("text", "")).strip().lower()
        conf = float(b.get("conf", 0.0))

        # Exact match
        if val == t:
            return conf

        # Partial match (value in OCR text or vice versa)
        if val in t or t in val:
            best = max(best, conf)

        # Fuzzy fallback: token overlap
        tokens = [tok for tok in val.split() if len(tok) > 2]
        if any(tok in t for tok in tokens):
            best = max(best, conf * 0.8)

    return best
