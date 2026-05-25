# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI application
app = FastAPI(title="Titanic Survival Predictor API")

# 2. Load the model file you just generated
model = joblib.load("titanic_model.pkl")

# 3. Define what the incoming data structure must look like
class Passenger(BaseModel):
    Pclass: int
    Sex: int  # 0 for male, 1 for female
    Age: float

@app.get("/")
def home():
    return {"status": "API is running. Go to /docs for the interactive UI."}

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert incoming JSON data to a Pandas DataFrame
    input_data = pd.DataFrame([passenger.model_dump()])
    
    # Run the prediction (0 = Died, 1 = Survived)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "survived_prediction": int(prediction),
        "survival_probability": float(probability)
    }