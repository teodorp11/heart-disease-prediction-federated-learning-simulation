import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(hospital_id: int):
    """
    Loads data for a specific hospital, performs scaling, 
    and returns features and target.
    """
    file_path = f"data/hospital_{hospital_id}.csv"
    df = pd.read_csv(file_path)
    
    # Define features
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def get_model_params(model):
    """Extracts parameters from a sklearn LogisticRegression model."""
    if model.fit_intercept:
        return [model.coef_, model.intercept_]
    return [model.coef_]

def set_model_params(model, params):
    """Sets parameters for a sklearn LogisticRegression model."""
    model.coef_ = np.array(params[0])
    if model.fit_intercept:
        model.intercept_ = np.array(params[1])
    return model