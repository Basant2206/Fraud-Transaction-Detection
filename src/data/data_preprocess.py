import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import joblib

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
  
    df = pd.read_csv(data_url)
    return df

def preprocess_data(train: pd.DataFrame, test: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """Preprocess the data by scaling numerical features."""
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Scale the numerical features
    train['Amount'] = scaler.fit_transform(train[['Amount']])
    test['Amount'] = scaler.transform(test[['Amount']])
    joblib.dump(scaler, scaler_path)
    
    return train, test


def save_data(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to a CSV file."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    # Load parameters from the params.yaml in the root directory
    params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
    
    train_data_path = params['data_ingestion']['train_data_path']
    test_data_path = params['data_ingestion']['test_data_path']

    train_data_preprocessed = params['data_preprocess']['train_data_path']
    test_data_preprocessed = params['data_preprocess']['test_data_path']
    scaler_path = params['model']['scaler_path']

    # Load the training and testing sets
    train_df = load_data(data_url=train_data_path)  
    test_df = load_data(data_url=test_data_path)

    # Preprocess the training and testing sets
    train_df, test_df = preprocess_data(train_df, test_df, scaler_path=scaler_path)

    # Save the training and testing sets
    save_data(train_df, output_path=train_data_preprocessed)
    save_data(test_df, output_path=test_data_preprocessed)
    print("Preprocessing completed. Scaler saved at:", scaler_path)


if __name__ == "__main__":
    main()