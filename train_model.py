"""
train_model.py
 Fake News Detection.
Trains a TF-IDF + LogisticRegression model using Fake.csv and Real.csv.
"""

# -----------------------------
# Standard library imports
# -----------------------------
import re
import os

# -----------------------------
# Third-party imports
# -----------------------------
import pickle
import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import TesseractError
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------------
# OPTIONAL: OCR – extract text from images and create Fake.csv / Real.csv
# -------------------------------------------------------
def build_csv_from_images():
    """
    Extracts text with OCR from the images inside dataset/real_images
    and dataset/fake_images folders and creates the files Fake.csv and Real.csv.
    """
    real_dir = "dataset/real_images"
    fake_dir = "dataset/fake_images"

    real_texts = []
    fake_texts = []

    # REAL folder
    if os.path.isdir(real_dir):
        for filename in os.listdir(real_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(real_dir, filename)
                print(f"OCR → REAL: {img_path}")
                try:
                    text = pytesseract.image_to_string(Image.open(img_path))
                    if text.strip():
                        real_texts.append(text.strip())
                except (IOError, TesseractError) as e:
                    print(f"⚠ OCR Error, skipped {filename}: {e}")

    # FAKE folder
    if os.path.isdir(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(fake_dir, filename)
                print(f"OCR → FAKE: {img_path}")
                try:
                    text = pytesseract.image_to_string(Image.open(img_path))
                    if text.strip():
                        fake_texts.append(text.strip())
                except (IOError, TesseractError) as e:
                    print(f"⚠ OCR Error, skipped {filename}: {e}")

    # Create CSV files
    pd.DataFrame({"text": real_texts}).to_csv("Real.csv", index=False)
    pd.DataFrame({"text": fake_texts}).to_csv("Fake.csv", index=False)

    print("\n OCR process completed!")
    print("✔ Real.csv created.")
    print("✔ Fake.csv created.\n")


# -------------------------------------------------------
# Text cleaning (MUST MATCH app.py)
# -------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean text exactly like app.py cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------
# Model training
# -------------------------------------------------------
def train_and_save(out_path: str = "model.pkl") -> None:
    """
    Train model using:
    - Fake.csv → FAKE
    - Real.csv → REAL
    Uses cleaned text to perfectly match app.py behavior.
    """

    print("📥 Loading datasets...")

    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("Real.csv")

    if "text" not in df_fake.columns or "text" not in df_real.columns:
        raise ValueError("Both CSV files MUST contain a 'text' column.")

    # Remove NaN rows
    df_fake = df_fake.dropna(subset=["text"])
    df_real = df_real.dropna(subset=["text"])

    # Apply labels
    df_fake["label"] = "FAKE"
    df_real["label"] = "REAL"

    # Clean text (same as app.py)
    df_fake["clean"] = df_fake["text"].astype(str).apply(clean_text)
    df_real["clean"] = df_real["text"].astype(str).apply(clean_text)

    # Combine
    df = pd.concat([df_fake[["clean", "label"]], df_real[["clean", "label"]]], ignore_index=True)

    # Rename for model
    df = df.rename(columns={"clean": "text"})

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    features = df["text"]
    labels = df["label"].astype(str).str.upper()

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # TF-IDF Vectorizer (fixed typo)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.75,
        max_features=30000,
        ngram_range=(1, 2)
    )

    x_train_vec = vectorizer.fit_transform(x_train)

    # Logistic Regression model
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )
    model.fit(x_train_vec, y_train)

    # Save model + vectorizer
    with open(out_path, "wb") as file:
        pickle.dump((vectorizer, model), file)

    # Evaluate accuracy
    x_test_vec = vectorizer.transform(x_test)
    accuracy = model.score(x_test_vec, y_test)

    print("===========================================")
    print("🎉 Training completed successfully!")
    print(f"📌 Model saved as: {out_path}")
    print("📊 Test Accuracy:", round(accuracy, 3))
    print("===========================================")


# -------------------------------------------------------
# Run training
# -------------------------------------------------------
if __name__ == "__main__":
    train_and_save()
