from fastapi.responses import FileResponse
from src.data_loader import preprocess_data
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


class PhysicalFeatures(BaseModel):
    KinEng: float
    PotEng: float
    Volume: float
    Step: int


app = FastAPI(title="Physical Predictor")

model = joblib.load("models/model.joblib")


@app.post("/predict")
def predict(data: PhysicalFeatures):
    df_raw = pd.DataFrame([data.dict()])

    try:
        X, _ = preprocess_data(df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing error: {e}")

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {"Кинетическая энергия": pred}


@app.get("/report")
def get_report():
    return FileResponse("predictions/report.html", media_type="text/html")
