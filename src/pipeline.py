# preprocessing pipeline

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
import os



def build_pipeline():
    numerical_columns = ["credit_score","age","tenure","balance","active_member","credit_card","products_number",
                         "estimated_salary"]
    categorical_colums = ["country","gender"]


    _pipeline = ColumnTransformer([('num_scale',StandardScaler(),numerical_columns),
                                ("one_hot_country",OneHotEncoder(drop='first'),['country']),
                                ("one_hot_gender",OneHotEncoder(),['gender'])])
    
    return _pipeline


def preprocess_and_save(df,pipeline):
    df = df.drop(columns = ["customer_id"])
    y = df['churn']
    X = df.drop(columns = ['churn'])

    X_transformed = pipeline.fit_transform(X)

    joblib.dump(pipeline, os.path.join("Bank Customer Churn", "models", "pipeline.pkl"))

    return X_transformed,y.values
