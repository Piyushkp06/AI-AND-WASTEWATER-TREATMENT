import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_anomaly_detector():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'processed_wwt_data.csv')
    model_save_path = os.path.join(script_dir, '..', '..', 'models', 'anomaly_detector_if.pkl')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature Selection for Monitoring
    # We want to monitor all critical inlet parameters to catch toxic shocks
    features = [
        'inlet_flow_rate_m3h', 
        'inlet_pH', 
        'inlet_temp_C', 
        'inlet_turbidity_NTU', 
        'inlet_cod_mgL',
        'COD_load_kg_h'
    ]
    
    X = df[features]
    
    print(f"Training Isolation Forest on {len(X)} historical samples...")
    # Initialize Isolation Forest
    # contamination parameter estimates the proportion of outliers in the data
    model = IsolationForest(
        n_estimators=100, 
        contamination=0.01, # Assume 1% of historical data might be anomalies
        random_state=42
    )
    
    # Train the model
    model.fit(X)
    
    # Evaluate on the training set just to see the outlier count
    predictions = model.predict(X)
    outliers = len(predictions[predictions == -1])
    
    print("-" * 30)
    print("Anomaly Detection Model Summary:")
    print(f"Total Samples: {len(X)}")
    print(f"Anomalies detected in historical data: {outliers} ({outliers/len(X)*100:.2f}%)")
    print("-" * 30)
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Anomaly detector saved to {model_save_path}")

if __name__ == "__main__":
    train_anomaly_detector()
