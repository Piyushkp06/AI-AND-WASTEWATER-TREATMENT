import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os

def train_chemical_dosing_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'processed_wwt_data.csv')
    model_save_path = os.path.join(script_dir, '..', '..', 'models', 'chemical_dosing_xgb.pkl')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature selection
    # For predicting optimal chemical dosage, we only know the INLET parameters
    features = [
        'inlet_flow_rate_m3h', 
        'inlet_pH', 
        'inlet_temp_C', 
        'inlet_turbidity_NTU', 
        'inlet_cod_mgL',
        'COD_load_kg_h'
    ]
    target = 'chemical_dosage_Lh'
    
    X = df[features]
    y = df[target]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training XGBoost Regressor on {len(X_train)} samples...")
    # Initialize the model
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        objective='reg:squarederror'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print("Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f} L/h")
    print(f"MAE:  {mae:.2f} L/h")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    
    # Display Feature Importance
    importance = model.feature_importances_
    for i,v in enumerate(importance):
        print(f"Feature: {features[i]:<25} Score: {v:.4f}")

if __name__ == "__main__":
    train_chemical_dosing_model()
