# This is the API for the credit card fraud detection system

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class TxIn(BaseModel):
    amount: float
    timestamp: str
    merchant_id: str
    device_change: int = 0

# Load models (assumes models/ directory mounted)
xgb = joblib.load('models/xgb_model.pkl')