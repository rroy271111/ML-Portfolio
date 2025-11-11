# Framework for deploying credit fraud detection system model

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

from features.feature_builder import build_features


# Load model path from environment or use default
MODEL_PATH = os.environ.get("MODEL_PATH", "src/models/saved_models/xgb_model.pkl")
model = None

app = FastAPI(title="Credit Fraud Detection API")


# Input schema for transaction
class TxIn(BaseModel):
    transaction_id: str
    card_token: str
    amount: float
    timestamp: str
    merchant_id: str
    device_id: str
    ip: str


@app.on_event("startup")
def load_model():
    """Load trained model into memory at startup"""
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


@app.post("/predict")
def predict(tx: TxIn):
    """Predict fraud probability for a given transaction"""
    try:
        # Convert input into a DataFrame
        df = pd.DataFrame([{
            "transaction_id": tx.transaction_id,
            "card_token": tx.card_token,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
            "merchant_id": tx.merchant_id,
            "device_id": tx.device_id,
            "ip": tx.ip
        }])

        # Feature engineering
        X = build_features(df)

        # Make prediction
        probs = model.predict_proba(X)[:, 1]
        return {
            "fraud_probability": float(probs[0]),
            "prediction": "FRAUD" if probs[0] > 0.5 else "LEGIT",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
