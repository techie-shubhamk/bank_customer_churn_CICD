<!-- ann-pytorch-churn/
│
├── data/
│   └── churn.csv
│
├── src/
│   ├── pipeline.py        # preprocessing pipeline
│   ├── dataset.py         # PyTorch Dataset
│   ├── model.py           # ANN model
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── main.py            # FastAPI
│
├── models/
│   ├── model.pth
│   └── scaler.pkl
│
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci-cd.yml -->
# bank_customer_churn_CICD
# Kaggle dataset for practice 
# docker pull your_username/churn-pytorch:latest
# docker run -p 8000:8000 your_username/churn-pytorch:latest
# http://localhost:8000

<!-- import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "credit_score": 650,
        "country": "France",
        "gender": "Male",
        "age": 40,
        "tenure": 3,
        "balance": 60000,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 50000
    }
)

print(response.json()) -->

# 

<!-- import sys
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR) -->
