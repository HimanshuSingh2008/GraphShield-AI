import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "backend" / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = ROOT / "ml" / "data" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "amount",
    "step",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

app = FastAPI(title="GraphShield AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    step: int = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "GraphShield AI API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    row = pd.DataFrame(
        [transaction.model_dump()],
        columns=FEATURES,
    )

    fraud_probability = float(model.predict_proba(row)[0][1])
    threshold = 0.5
    prediction = int(fraud_probability >= threshold)

    # Save transaction in Supabase
    transaction_response = (
        supabase_client.table("transactions")
        .insert(
            {
                "amount": transaction.amount,
                "transaction_time": datetime.now(timezone.utc).isoformat(),
                "is_fraud": bool(prediction),
            }
        )
        .execute()
    )

    response_data = transaction_response.data
    if not isinstance(response_data, list) or not response_data:
        raise RuntimeError("Supabase did not return the saved transaction")

    transaction_row = response_data[0]
    if not isinstance(transaction_row, dict) or "id" not in transaction_row:
        raise RuntimeError("Supabase response is missing the transaction ID")

    transaction_id = transaction_row["id"]

    # Save prediction in Supabase
    supabase_client.table("fraud_predictions").insert(
        {
            "transaction_id": transaction_id,
            "fraud_probability": fraud_probability,
            "predicted_label": prediction,
            "model_version": "random_forest_v1",
        }
    ).execute()

    return {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "label": "Fraud" if prediction else "Legitimate",
        "threshold": threshold,
        "database_status": "Saved successfully",
    }
