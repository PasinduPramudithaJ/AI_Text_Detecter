import os
import torch
import numpy as np
import json
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
from docx import Document
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pdfkit

# ---------- FLASK SETUP ----------
app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- MODEL SETUP ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()

# ---------- TEXT EXTRACTION ----------
def extract_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append({"page": i + 1, "text": page.extract_text() or ""})
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
    if skip_indices is None:
        skip_indices = []

    results = []
    ai_probs = []

    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            if len(p) < 25:
                continue
            if (page["page"], idx) in skip_indices:
                continue
            ppl = sentence_perplexity(p)
            ai_prob = max(0, min(1, (90 - ppl) / 90))
            human_prob = 1 - ai_prob
            ai_probs.append(ai_prob)
            results.append({
                "page": page["page"],
                "index": idx,
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

        skip_raw = request.form.getlist("skip")
        skip_indices = [tuple(map(int, x.split("-"))) for x in skip_raw] if skip_raw else []

        ai, human, details = analyze_paragraphs(pages, skip_indices)

        analysis_file = os.path.join(UPLOAD_DIR, "latest_analysis.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump({"ai": ai, "human": human, "details": details}, f)

        return render_template("index.html", ai_score=ai, human_score=human, details=details, file_name=file.filename)

    return render_template("index.html")
# ---------- RUN ----------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
