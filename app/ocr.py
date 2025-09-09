import io
from typing import List, Dict
from PIL import Image
import numpy as np
import os

# Choose OCR engine (currently only EasyOCR implemented)
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")

# Lazy import to keep startup light
_reader = None


def get_easyocr_reader():
    """
    Initialize and return a cached EasyOCR reader.
    """
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)  # load English reader
    return _reader


def image_from_bytes(data: bytes) -> Image.Image:
    """
    Convert raw bytes into a PIL Image.
    """
    return Image.open(io.BytesIO(data)).convert("RGB")


def pdf_bytes_to_images(data: bytes, dpi: int = 300) -> List[Image.Image]:
    """
    Convert PDF bytes into a list of PIL images.
    """
    from pdf2image import convert_from_bytes
    pages = convert_from_bytes(data, dpi=dpi)
    return [p.convert("RGB") for p in pages]


def ocr_image_pil(img: Image.Image) -> List[Dict]:
    """
    Run OCR on a single PIL image and return a list of blocks:
    {
      "text": recognized text,
      "conf": confidence score,
      "bbox": [x_min, y_min, x_max, y_max]
    }
    """
    reader = get_easyocr_reader()
    arr = np.array(img)
    results = reader.readtext(arr)

    blocks = []
    for bbox, text, conf in results:
        # easyocr bbox is 4 points -> flatten to min/max
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        blocks.append({
            "text": text,
            "conf": float(conf),
            "bbox": [x_min, y_min, x_max, y_max]
        })
    return blocks


def run_ocr(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Main OCR entrypoint.
    - Handles both images and PDFs
    - Returns combined blocks across pages (for PDFs)
    """
    if filename.lower().endswith(".pdf"):
        pages = pdf_bytes_to_images(file_bytes)
        all_blocks = []
        for p in pages:
            all_blocks.extend(ocr_image_pil(p))
        return all_blocks
    else:
        img = image_from_bytes(file_bytes)
        return ocr_image_pil(img)
