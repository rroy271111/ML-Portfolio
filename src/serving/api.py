# This is the API for the credit card fraud detection system

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Any
import os

from features.feature_builder import build_features

MODEL_PATH = os.environ.get("MODEL_PATH", "src/models/saved_models/xgb_model.pkl")
model = None

app = FastAPI(title="Credit Fraud API")

class TxIn(BaseModel):
    amount: float
    timestamp: str
    merchant_id: str
    device_change: int = 0

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(tx: TxIn):
    # Build a DataFrame matching feature_builder expectations
    df = pd.DataFrame([{
        "amount": tx.amount,
        "timestamp": tx.timestamp,
        "merchant_id": tx.merchant_id
    }])

    # Use feature builder
    X = build_features(df)
    try:
        probs = model.predict_proba(X)[:,1]
        return {"fraud_prob": float(probs[0])}
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))

