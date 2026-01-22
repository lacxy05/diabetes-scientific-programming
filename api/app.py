import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")

dire = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pipeline_path = os.path.join(dire, "models", "pipeline.pkl")
meta_path = os.path.join(dire, "models", "meta.pkl")

pipeline = joblib.load(pipeline_path)
meta = joblib.load(meta_path)

best_threshold = meta["threshold"]["best_threshold"]

feature = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict_diabetes(patient: PatientData):
    df = pd.DataFrame(
        [[getattr(patient, col) for col in feature]],
        columns=feature
    )

    
    prob = pipeline.predict_proba(df)[0][1]
    prediction = "Diabetes" if prob >= best_threshold else "No Diabetes"

    return {
        "diagnosis": prediction,
        "probability": round(float(prob), 2),
        "threshold_used": best_threshold
    }

@app.get("/")
def root():
    return {"message": "API is running"}