import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")

dire = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_aux = os.path.join(dire, "models", "model.pkl")
scaler_aux = os.path.join(dire, "models", "scaler.pkl")

model = joblib.load(model_aux)
scaler = joblib.load(scaler_aux)


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
    # Crear DataFrame en el orden correcto
    df = pd.DataFrame(
        [[getattr(patient, col) for col in feature]],
        columns=feature
    )

   
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature
    )

    
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = "Diabetes" if prob >= 0.5 else "No Diabetes"

    return {
        "diagnosis": prediction,
        "probability": round(float(prob), 2)
    }
