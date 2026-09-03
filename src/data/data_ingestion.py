import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
  
    df = pd.read_csv(data_url)
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """Split the data into training and testing sets."""
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Class'])
    return train_df, test_df

def save_data(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to a CSV file."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    # Load parameters from the params.yaml in the root directory
    params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
    test_size = params['data_ingestion']['test_size']
    raw_data_path = params['data_ingestion']['raw_data_path']
    train_data_path = params['data_ingestion']['train_data_path']
    test_data_path = params['data_ingestion']['test_data_path']

    # Load data from the specified URL
    df = load_data(data_url=raw_data_path)

    # Split the data into training and testing sets
    train_df, test_df = split_data(df, test_size=test_size, random_state=42)

    # Save the training and testing sets
    save_data(train_df, output_path=train_data_path)
    save_data(test_df, output_path=test_data_path)
    print(test_size, raw_data_path, train_data_path, test_data_path)


if __name__ == "__main__":
    main()