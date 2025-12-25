import os
import torch
import numpy as np
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from docx import Document
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------- FLASK SETUP ----------
app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- MODEL SETUP ----------
device = torch.device("cpu")  # change to "cuda" if GPU available

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()

# ---------- TEXT EXTRACTION ----------
def extract_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append({
            "page": i + 1,
            "text": page.extract_text() or ""
        })
    return pages

def extract_docx(path):
    doc = Document(path)
    pages, buffer, page_no = [], "", 1
    for para in doc.paragraphs:
        buffer += para.text + "\n"
        if len(buffer) > 900:
            pages.append({"page": page_no, "text": buffer})
            buffer, page_no = "", page_no + 1
    if buffer.strip():
        pages.append({"page": page_no, "text": buffer})
    return pages

# ---------- AI DETECTION ----------
def sentence_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def analyze_paragraphs(pages, skip_indices=None):
    """
    Analyze paragraphs with optional skipping
    skip_indices: list of tuples (page_number, paragraph_index)
    """
    if skip_indices is None:
        skip_indices = []

    results = []
    ai_probs = []

    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            if len(p) < 25:  # skip too short text automatically
                continue
            if (page["page"], idx) in skip_indices:
                continue  # skip manually marked paragraph

            ppl = sentence_perplexity(p)
            ai_prob = max(0, min(1, (90 - ppl) / 90))
            human_prob = 1 - ai_prob
            ai_probs.append(ai_prob)

            results.append({
                "page": page["page"],
                "index": idx,  # paragraph index for skipping
                "text": p,
                "ai": round(ai_prob * 100, 2),
                "human": round(human_prob * 100, 2),
                "label": "AI-generated" if ai_prob > 0.6 else "Human-written"
            })

    overall_ai = round(np.mean(ai_probs) * 100, 2) if ai_probs else 0
    overall_human = round(100 - overall_ai, 2)
    return overall_ai, overall_human, results

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded"

        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)

        ext = file.filename.lower()
        if ext.endswith(".pdf"):
            pages = extract_pdf(path)
        elif ext.endswith(".docx"):
            pages = extract_docx(path)
        else:
            return "Unsupported file type"

        # Check for skipped paragraphs submitted via form
        skip_raw = request.form.getlist("skip")
        skip_indices = [tuple(map(int, x.split("-"))) for x in skip_raw] if skip_raw else []

        ai, human, details = analyze_paragraphs(pages, skip_indices)

        return render_template(
            "index.html",
            ai_score=ai,
            human_score=human,
            details=details
        )

    return render_template("index.html")
# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
