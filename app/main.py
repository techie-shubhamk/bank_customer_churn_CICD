# fast API code

import sys
import os
BASE_DIR = "/Users/shubham/Desktop/ML_REPO/Bank Customer Churn"
sys.path.append(BASE_DIR)


from fastapi import FastAPI


from src.predict import predict
import pandas as pd



app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "PyTorch Churn Model API"}

@app.post("/predict")
def get_prediction(data: dict):
    
    input_data = pd.DataFrame([list(data.values())], columns=list(data.keys()))
    result = predict(input_data)

    return {"churn_probability": result}