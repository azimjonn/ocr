# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests, fitz, tempfile, os
from paddleocr import PaddleOCR

app = FastAPI()

# ── CONFIG ────────────────────────────────────────────────────────────────
LANG  = "uz"          # change if the document language is not English
DPI   = 300           # rendering resolution; 300 is safe for small fonts
GPU   = False          # False → CPU only
ocr   = PaddleOCR(use_gpu=GPU, lang=LANG, use_angle_cls=False, drop_score=0.05)
# drop_score lowered: keeps low-confidence glyphs instead of discarding

# ── UTILITY  FUNCTIONS ────────────────────────────────────────────────────
def _as_str(x):
    if isinstance(x, (list, tuple)):   # Paddle returns list of chars for some langs
        return "".join(map(str, x))
    return str(x)

def _ocr_image(path: str) -> str:
    """
    Runs OCR on a single image file, returns full page text.
    Accepts both old (2-fields) and new (3-fields) PaddleOCR line formats.
    """
    result = ocr.ocr(path, cls=False)
    if not result:
        return ""
    # result[0] = list of detected lines for this image
    lines = []
    for item in result[0]:
        if len(item) == 2:               # [bbox, (text, conf)]
            _, (txt, _) = item
        elif len(item) == 3:             # [bbox, text, conf]
            _, txt, _   = item
        else:
            continue
        lines.append(_as_str(txt))
    return "\n".join(lines)

# ── ENDPOINT ──────────────────────────────────────────────────────────────
@app.post("/ocr")
def extract(url: str):
    # 1. download PDF
    try:
        pdf_bytes = requests.get(url, timeout=15).content
    except Exception as e:
        raise HTTPException(400, f"download error: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        pdf_path = f.name

    # 2. open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        os.remove(pdf_path)
        raise HTTPException(400, f"invalid PDF: {e}")

    # 3. iterate pages → OCR
    pages_text = []
    for idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=DPI)
        img_path = f"/tmp/_p{idx}.png"
        pix.save(img_path)
        pages_text.append(_ocr_image(img_path))
        os.remove(img_path)

    doc.close()
    os.remove(pdf_path)
    return {"pages": pages_text}
