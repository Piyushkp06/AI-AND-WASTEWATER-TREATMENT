import pandas as pd
import numpy as np
import os

def preprocess_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'synthetic_wwt_data.csv')
    output_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'processed_wwt_data.csv')
    
    print(f"Loading raw data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find raw data at {input_path}")
        return
        
    print(f"Raw data shape: {df.shape}")
    
    # 1. Handle Missing Values
    # Since this is time series data, forward fill is usually appropriate, followed by backward fill
    # However, since we just injected random NaNs, simple median imputation per column is also safe
    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values. Imputing with median...")
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
    
    # 2. Feature Engineering
    # Calculate Organic Load (COD load) in kg/h
    # Formula: Flow (m3/h) * COD (mg/L) / 1000 = kg/h
    if 'inlet_flow_rate_m3h' in df.columns and 'inlet_cod_mgL' in df.columns:
        df['COD_load_kg_h'] = (df['inlet_flow_rate_m3h'] * df['inlet_cod_mgL']) / 1000.0
        print("Engineered new feature: 'COD_load_kg_h'")

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} preprocessed samples to {output_path}")
    print(df.head())

if __name__ == "__main__":
    preprocess_data()
