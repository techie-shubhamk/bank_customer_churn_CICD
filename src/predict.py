import sys
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR)



# Get the directory where the app is located

# MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")



import torch
import torch.nn as nn
from src.model import bank_churn_model
import joblib
import pandas as pd







# def predict(input_data):
#     pipeline = joblib.load("Bank Customer Churn/models/pipeline.pkl")
#     X = pipeline.transform([input_data])
#     # model = bank_churn_model(input_dim = X.shape[1],output_dim=1,num_of_layer=2,num_nurone_perlayer=99,dropout=0.309)
#     model = bank_churn_model(input_dim=X.shape[1])
#     model.load_state_dict(torch.load("Bank Customer Churn/models/model.pt"))
#     model.eval()

#     with torch.no_grad():
#         prediction = model(torch.tensor(X, dtype=torch.float32))


#     return float(prediction.item())



pipeline_path = os.path.join(BASE_DIR, "models", "pipeline.pkl")
pipeline = joblib.load(pipeline_path)
# Load once


# dummy_input = pipeline.transform([[650, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
# input_dim = dummy_input.shape[1]




def predict(input_data):

    
    X = pipeline.transform(input_data)
    # model = bank_churn_model(input_dim= X.shape[1])
    model = bank_churn_model(
                                input_dim= X.shape[1], 
                                output_dim=1,
                                num_of_layer=2,
                                num_nurone_perlayer=99,
                                dropout=0.309
                            )
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "model.pt")))
    model.eval()

    with torch.no_grad():
        prediction = model(torch.tensor(X, dtype=torch.float32))

    return (prediction.squeeze() > 0.5).int()


if __name__ == "__main__":
    # Example input data
    input_data = pd.DataFrame([[650, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000],
                               [608, 'Spain', 'Female', 41, 1, 83807.86, 1, 0, 1, 112542.58]],
                              columns=["credit_score","country","gender","age","tenure","balance",
                                       "products_number","active_member","credit_card","estimated_salary"])
    result = predict(input_data)
    print(f"Prediction: {result}")