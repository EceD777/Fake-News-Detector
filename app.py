"""
Fake News Detection Web App 

Features Included:
✔ ML fallback (model.pkl)
✔ Enhanced real check
✔ Short Text Booster
✔ Expanded trusted domain list
✔ Strong DuckDuckGo search verification
✔ OCR + URL article extraction
✔ Extended API (text, URL, image)
✔ Pylint-clean (no broad excepts, correct import order, docstrings)
"""

# ----------------------------
# IMPORTS (Correct Pylint order)
# ----------------------------
from urllib.parse import urlparse
from typing import Optional
import pickle
import re

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
)
from newspaper import Article
from newspaper.article import ArticleException
from pytesseract import image_to_string, TesseractError
from PIL import Image
from duckduckgo_search import DDGS


# ----------------------------
# FLASK APP
# ----------------------------
app = Flask(__name__)
app.secret_key = "SECRET_KEY_FOR_SESSION"


# ----------------------------
# LOAD ML MODEL
# ----------------------------
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "rb") as f:
    VECTORIZER, MODEL = pickle.load(f)


# ----------------------------
# TRUSTED DOMAIN LIST
# ----------------------------
TRUSTED_DOMAINS = [
    # short forms
    "bbc", "cnn", "reuters", "apnews", "guardian",
    "cbc", "globalnews", "bloomberg", "nbc",
    "skynews", "euronews", "aljazeera", "forbes",
    "politico", "economist", "foxnews", "abcnews",
    "cbsnews", "usatoday", "nypost", "latimes",
    "wsj", "washingtonpost", "nytimes",

    # expanded forms
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com",
    "apnews.com", "theguardian.com", "cbc.ca",
    "globalnews.ca", "bloomberg.com", "nbcnews.com",
    "euronews.com", "aljazeera.com", "forbes.com",
    "politico.com", "economist.com", "foxnews.com",
    "abcnews.go.com", "cbsnews.com", "usatoday.com",
    "nypost.com", "latimes.com", "wsj.com",
    "washingtonpost.com", "nytimes.com",
    "telegraph.co.uk", "independent.co.uk",
    "sciencedaily.com", "nature.com",
    "scientificamerican.com",
]

TRUSTED_KEYWORDS = [
    "reuters", "bbc", "scientific american",
    "science daily", "medical news today",
    "live science", "nature", "sciencedaily",
]


# ----------------------------
# TRUSTED DOMAIN CHECKS
# ----------------------------
def is_domain_trusted(url: str) -> bool:
    """Return True if URL belongs to a trusted domain."""
    domain = urlparse(url).netloc.lower()
    return any(t in domain for t in TRUSTED_DOMAINS)


def keyword_trusted_text(text: str) -> bool:
    """Return True if trusted keyword appears in text."""
    text_low = text.lower()
    return any(k in text_low for k in TRUSTED_KEYWORDS)


def domain_in_text(text: str) -> bool:
    """Return True if trusted domain appears inside text."""
    text_low = text.lower()
    return any(d in text_low for d in TRUSTED_DOMAINS)


# ----------------------------
# MATCH TRAINING CLEANING
# ----------------------------
def clean_text(text: str) -> str:
    """Clean text exactly like in training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# LOAD CSV OVERRIDE DATASETS (CLEANED)
# ----------------------------
DF_FAKE = pd.read_csv("Fake.csv")
DF_REAL = pd.read_csv("Real.csv")

DF_FAKE["clean"] = DF_FAKE["text"].astype(str).apply(clean_text)
DF_REAL["clean"] = DF_REAL["text"].astype(str).apply(clean_text)

FAKE_SET = set(DF_FAKE["clean"])
REAL_SET = set(DF_REAL["clean"])


# ----------------------------
# SEARCH VERIFICATION
# ----------------------------
def search_verification(query: str) -> bool:
    """Check DuckDuckGo search results for trusted domains."""
    try:
        with DDGS() as dd:
            results = dd.text(query, max_results=20)
    except (OSError, TimeoutError, ValueError):
        return False

    for r in results:
        url = r.get("href", "")
        if url and is_domain_trusted(url):
            return True
    return False


# ----------------------------
# SHORT TEXT BOOSTER
# ----------------------------
def short_text_booster(text: str) -> bool:
    """Boost REAL detection for headlines / short posts."""
    words = len(text.split())

    if words <= 200:
        if domain_in_text(text):
            return True
        if keyword_trusted_text(text):
            return True
        if search_verification(text):
            return True

    return False


# ----------------------------
# ENHANCED REAL CHECK
# ----------------------------
def enhanced_real_check(text: str, url: Optional[str]) -> bool:
    """Powerful verification method combining multiple signals."""
    if url and is_domain_trusted(url):
        return True
    if keyword_trusted_text(text):
        return True
    if domain_in_text(text):
        return True
    if short_text_booster(text):
        return True
    if search_verification(text[:200]):
        return True
    return False


# ----------------------------
# HOME PAGE
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    """Render index.html with optional result."""
    return render_template("index.html", result=session.pop("result", None))


# ----------------------------
# PREDICT (HTML FORM)
# ----------------------------
@app.route("/", methods=["POST"])
def predict():
    """Main prediction logic with CSV override + ML fallback."""

    text = ""
    url_used = None

    # Inputs
    news_text = request.form.get("news_text", "").strip()
    news_link = request.form.get("news_link", "").strip()
    news_file = request.files.get("news_file")

    # OCR Upload
    if news_file and news_file.filename:
        try:
            img = Image.open(news_file)
            text = image_to_string(img).strip()
        except (OSError, TesseractError):
            session["result"] = "❌ Could not process image OCR."
            return redirect(url_for("home"))

    # URL Extraction
    elif news_link:
        url_used = news_link
        try:
            article = Article(news_link)
            article.download()
            article.parse()
            text = article.text.strip()
        except ArticleException:
            session["result"] = "❌ Could not extract article."
            return redirect(url_for("home"))

    # Plain text
    else:
        text = news_text

    if not text:
        session["result"] = "⚠️ No input provided."
        return redirect(url_for("home"))

    # Clean input for CSV override + model
    clean_input = clean_text(text)

    # ----------------------------
    # CSV OVERRIDE (guaranteed correct)
    # ----------------------------
    if clean_input in FAKE_SET:
        session["result"] = "🔴 FAKE NEWS (CSV Verified)"
        return redirect(url_for("home"))

    if clean_input in REAL_SET:
        session["result"] = "🟢 REAL NEWS (CSV Verified)"
        return redirect(url_for("home"))

    # ----------------------------
    # ML FALLBACK
    # ----------------------------
    vec = VECTORIZER.transform([clean_input])
    prediction = MODEL.predict(vec)[0]

    result = "🟢 REAL NEWS" if prediction == "REAL" else "🔴 FAKE NEWS"

    # Enhanced verification
    if enhanced_real_check(text, url_used):
        result = "🟢 REAL NEWS (Verified)"

    session["result"] = result
    return redirect(url_for("home"))


# ----------------------------
# EXTENDED API
# ----------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Full JSON API with CSV override, ML fallback, and verification."""

    text = ""
    url_used: Optional[str] = None

    text_input = request.form.get("text", "").strip()
    url_input = request.form.get("url", "").strip()
    image_file = request.files.get("image")

    # OCR input
    if image_file and image_file.filename:
        try:
            img = Image.open(image_file)
            text = image_to_string(img).strip()
        except (OSError, TesseractError):
            return jsonify({"error": "OCR failed"}), 400

    # URL input
    elif url_input:
        url_used = url_input
        try:
            article = Article(url_input)
            article.download()
            article.parse()
            text = article.text.strip()
        except ArticleException:
            return jsonify({"error": "URL extraction failed"}), 400

    # text input
    elif text_input:
        text = text_input

    else:
        return jsonify({"error": "No input provided"}), 400

    # Clean input for CSV override + model
    clean_input = clean_text(text)

    # CSV override
    if clean_input in FAKE_SET:
        return jsonify({
            "prediction": "FAKE",
            "override": True,
            "overrideType": "CSV_FAKE_MATCH",
            "verified": False,
            "source": "Fake.csv",
            "textLength": len(text.split())
        })

    if clean_input in REAL_SET:
        return jsonify({
            "prediction": "REAL",
            "override": True,
            "overrideType": "CSV_REAL_MATCH",
            "verified": False,
            "source": "Real.csv",
            "textLength": len(text.split())
        })

    # ML fallback
    vec = VECTORIZER.transform([clean_input])
    prediction = MODEL.predict(vec)[0]

    verified = enhanced_real_check(text, url_used)

    return jsonify({
        "prediction": prediction,
        "override": False,
        "verified": verified,
        "source": "ML_MODEL",
        "textLength": len(text.split()),
        "urlUsed": url_used
    })


# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
