#!/usr/bin/env python3
"""
Scam Detection System using DistilBERT
---------------------------------------
Train: python scam_detector.py --mode train --data scam_detection_dataset.csv
API:   python scam_detector.py --mode api
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR   = "./saved_scam_model"
BASE_MODEL  = "distilbert-base-uncased"
MAX_LENGTH  = 128
EPOCHS      = 20
BATCH_SIZE  = 8

# ─────────────────────────────────────────────
# EXTRA TRAINING SAMPLES (injected at runtime)
# ─────────────────────────────────────────────
EXTRA_LEGIT = [
    "Your OTP for SBI login is 293847. Valid for 10 minutes. Do NOT share with anyone.",
    "Your OTP for ICICI Bank transaction is 748291. Do not share this with anyone.",
    "Your OTP for UPI payment is 938471. Valid for 5 minutes. Never share OTP with anyone.",
    "OTP for your Paytm login is 384729. It is valid for 5 mins. Do not share with anyone.",
    "Your OTP for Google account verification is 192837. Do not share it with anyone.",
    "HDFC Bank: OTP 736281 for your net banking login. Valid for 5 mins. Do not share.",
    "Your OTP for PhonePe transaction of Rs.500 is 829374. Valid for 10 minutes.",
    "Axis Bank OTP for login: 473829. Do not share this OTP with anyone including bank staff.",
    "Your OTP for HDFC Bank login is 847362. Valid for 5 minutes. Do NOT share with anyone.",
]

EXTRA_SCAM = [
    "Your OTP is 482910. Never share this. But to verify account click: bit.ly/secure-bank",
    "URGENT: Your OTP has expired. Generate new OTP by clicking here: hdfc-secure-otp.com",
    "Share your OTP with our agent to complete KYC verification immediately.",
    "Enter OTP received on mobile to claim your Rs.10,000 reward: fake-bank-otp.com",
]


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy" : round(acc, 4),
        "f1"       : round(f1, 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
    }


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(data_path: str):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(data_path)

    # Inject extra samples
    extra_rows = (
        [{"text": t, "label": 0, "label_name": "legitimate"} for t in EXTRA_LEGIT] +
        [{"text": t, "label": 1, "label_name": "scam"}       for t in EXTRA_SCAM]
    )
    df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    print(f"✅ Total samples: {len(df)} | Scam: {len(df[df.label==1])} | Legit: {len(df[df.label==0])}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    train_dataset = ScamDataset(train_df["text"].values, train_df["label"].values, tokenizer)
    test_dataset  = ScamDataset(test_df["text"].values,  test_df["label"].values, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️  Training on: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir                  = "./scam_model_output",
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        warmup_steps                = 10,
        weight_decay                = 0.01,
        logging_dir                 = "./logs",
        logging_steps               = 5,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
    )

    print("\n🚀 Starting training...\n")
    trainer.train()

    results = trainer.evaluate()
    print("\n===== Evaluation Results =====")
    print(f"  Accuracy  : {results['eval_accuracy']  * 100:.2f}%")
    print(f"  F1 Score  : {results['eval_f1']        * 100:.2f}%")
    print(f"  Precision : {results['eval_precision'] * 100:.2f}%")
    print(f"  Recall    : {results['eval_recall']    * 100:.2f}%")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n💾 Model saved to {MODEL_DIR}")


# ─────────────────────────────────────────────
# PREDICTOR CLASS (shared by CLI + API)
# ─────────────────────────────────────────────
class ScamPredictor:
    def __init__(self):
        # Check if saved model exists, otherwise fall back to base model
        if os.path.exists(MODEL_DIR):
            print(f"✅ Loading custom model from {MODEL_DIR}")
            load_path = MODEL_DIR
        else:
            print(f"⚠️ Custom model not found. Using base model: {BASE_MODEL}")
            load_path = BASE_MODEL
            
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
        self.model     = DistilBertForSequenceClassification.from_pretrained(load_path)
        self.model     = self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        return {
            "label"       : "scam" if prediction == 1 else "legitimate",
            "is_scam"     : bool(prediction == 1),
            "confidence"  : round(float(probs[prediction]) * 100, 2),
            "scam_prob"   : round(float(probs[1]) * 100, 2),
            "legit_prob"  : round(float(probs[0]) * 100, 2),
        }

# ─────────────────────────────────────────────
# FASTAPI — GLOBAL INSTANCE FOR RENDER/PRODUCTION
# ─────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Nexus AI Scam Detector API", version="1.0.0")
predictor = None # Initialize only when needed to save memory

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Nexus AI Scam Detector API is running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: MessageRequest):
    global predictor
    if predictor is None:
        predictor = ScamPredictor()
    
    if not request.text.strip():
        return {"error": "Empty message provided"}
    return predictor.predict(request.text)

def run_api():
    import uvicorn
    print("\n🌐 API running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ─────────────────────────────────────────────
# CLI PREDICT MODE
# ─────────────────────────────────────────────
def run_cli():
    predictor = ScamPredictor()
    print("\n🔍 Scam Detector — Type a message to check (Ctrl+C to quit)\n")
    while True:
        try:
            text   = input("Enter message: ").strip()
            if not text:
                continue
            result = predictor.predict(text)
            label  = "🚨 SCAM" if result["is_scam"] else "✅ LEGITIMATE"
            print(f"  Result     : {label}")
            print(f"  Confidence : {result['confidence']}%")
            print(f"  Scam prob  : {result['scam_prob']}%")
            print(f"  Legit prob : {result['legit_prob']}%\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scam Detector using DistilBERT")
    parser.add_argument(
        "--mode",
        choices=["train", "api", "cli"],
        required=True,
        help="train | api | cli"
    )
    parser.add_argument(
        "--data",
        default="scam_detection_dataset.csv",
        help="Path to CSV dataset (required for train mode)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data)
    elif args.mode == "api":
        run_api()
    elif args.mode == "cli":
        run_cli()
