import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_aeration_control_model():
    data_path = '../../data/raw/synthetic_wwt_data.csv'
    model_save_path = '../../models/aeration_rf_model.pkl'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature Selection for Aeration Control
    # We want the model to recommend the optimal Blower RPM given the incoming load
    # and a target Dissolved Oxygen (DO). We'll use the historical DO as the 'target_DO' feature.
    
    features = [
        'inlet_flow_rate_m3h', 
        'inlet_cod_mgL', 
        'effluent_do_mgL' # Treated as Target DO for the recommendation
    ]
    target = 'aeration_blower_rpm'
    
    X = df[features]
    y = df[target]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest Regressor on {len(X_train)} samples...")
    # Initialize a Random Forest model (more robust for control recommendations)
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=8, 
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print("Aeration Model Evaluation:")
    print(f"RMSE: {rmse:.2f} RPM")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Aeration control model saved to {model_save_path}")
    
if __name__ == "__main__":
    train_aeration_control_model()
