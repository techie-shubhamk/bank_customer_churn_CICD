ann-pytorch-churn/
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
└── .github/workflows/ci-cd.yml
# bank_customer_churn_CICD
Kaggle dataset for practice 
