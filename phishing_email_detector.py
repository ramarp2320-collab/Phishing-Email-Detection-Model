import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("phishing_emails.csv")
KEYWORDS = (
    "verify", "urgent", "password", "account", "bank", "login",
    "click here", "update", "suspended", "winner", "prize",
    "confirm", "limited time", "security alert", "payment"
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def normalize_label(value):
    label = str(value).strip().lower()
    if label in {"phishing", "spam", "1", "1.0", "malicious"}:
        return "Phishing"
    if label in {"safe", "ham", "0", "0.0", "legitimate", "legit"}:
        return "Safe"
    raise ValueError(f"Unknown label: {value}. Use Phishing/Safe, spam/ham, or 1/0.")


def extra_features(texts):
    rows = []
    for item in texts:
        text = str(item)
        lower = text.lower()
        rows.append([
            len(URL_RE.findall(text)),
            sum(lower.count(word) for word in KEYWORDS),
            len(text)
        ])
    return np.asarray(rows, dtype=float)


def main():
    if not DATA_FILE.exists():
        print("Dataset not found. Create phishing_emails.csv with email_text,label columns.")
        sys.exit(1)

    data = pd.read_csv(DATA_FILE)
    if not {"email_text", "label"}.issubset(data.columns):
        raise ValueError("CSV must contain email_text and label columns.")

    data = data.dropna(subset=["email_text", "label"]).copy()
    data["email_text"] = data["email_text"].astype(str)
    data["label"] = data["label"].apply(normalize_label)

    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Provide at least two examples for each class: Phishing and Safe.")

    X = data["email_text"].to_numpy()
    y = data["label"].to_numpy()

    # Split first to prevent data leakage during feature fitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    train_text = vectorizer.fit_transform(X_train)
    test_text = vectorizer.transform(X_test)

    scaler = StandardScaler()
    train_extra = scaler.fit_transform(extra_features(X_train))
    test_extra = scaler.transform(extra_features(X_test))

    train_all = hstack([train_text, train_extra]).tocsr()
    test_all = hstack([test_text, test_extra]).tocsr()

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_all, y_train)
    predictions = model.predict(test_all)

    print("\\n===== PHISHING EMAIL DETECTION RESULTS =====")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["Phishing", "Safe"], cmap="Blues"
    )
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved as confusion_matrix.png")

    joblib.dump(
        {"vectorizer": vectorizer, "scaler": scaler, "model": model},
        "phishing_email_model.joblib"
    )
    print("Model saved as phishing_email_model.joblib")

    new_email = input("\\nEnter an email to test (or press Enter to finish): ").strip()
    if new_email:
        new_text = vectorizer.transform([new_email])
        new_extra = scaler.transform(extra_features([new_email]))
        result = model.predict(hstack([new_text, new_extra]).tocsr())[0]
        print("Prediction:", result)
        print("Note: This educational text-based prediction is not a safety guarantee.")


if __name__ == "__main__":
    main()
