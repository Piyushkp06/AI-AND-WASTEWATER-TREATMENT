import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_wwt_data(num_samples=10000):
    # Resolve path relative to this script's location to be robust regardless of where it is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'synthetic_wwt_data.csv')
    
    np.random.seed(42)
    
    # Time series over several months
    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
    
    # 1. Simulate Inlet Water Parameters
    # Base flow with some diurnal variation
    base_flow = 8000 # m3/day roughly -> 333 m3/h
    diurnal_pattern = np.sin(np.linspace(0, 24 * np.pi * (num_samples/24), num_samples)) * 50
    inlet_flow_rate = np.random.normal(333, 20, num_samples) + diurnal_pattern
    
    # Industrial pH tends to be acidic or highly variable (e.g., from pickling/HSM)
    inlet_pH = np.random.normal(5.5, 1.2, num_samples)
    inlet_pH = np.clip(inlet_pH, 2.0, 11.0) 
    
    inlet_temp = np.random.normal(32, 5, num_samples) # Industrial water is often warm
    inlet_turbidity = np.random.normal(150, 40, num_samples)
    inlet_cod = np.random.normal(1200, 300, num_samples)
    
    # 2. Simulate Operational Controls (What the AI will eventually optimize)
    # The baseline operators apply chemical dosage (e.g., lime) inversely proportional to pH to neutralize it
    # We add some human error/noise
    chemical_dosage = np.where(inlet_pH < 7.0, (7.0 - inlet_pH) * 15, 0)
    chemical_dosage += np.random.normal(5, 2, num_samples) # Base dosage + noise
    chemical_dosage = np.clip(chemical_dosage, 0, 100) # L/hr or kg/hr
    
    # Blower RPM (Aeration) based roughly on flow and COD load
    aeration_blower_rpm = (inlet_flow_rate * 2) + (inlet_cod * 0.5) + np.random.normal(100, 50, num_samples)
    aeration_blower_rpm = np.clip(aeration_blower_rpm, 500, 3000)
    
    # 3. Simulate Effluent Parameters (The outcome)
    # If dosed correctly, pH approaches 7.2. 
    effluent_pH = inlet_pH + (chemical_dosage * 0.08)
    effluent_pH += np.random.normal(0, 0.3, num_samples) # Process noise
    
    # DO depends on blower speed and COD
    effluent_do = (aeration_blower_rpm / 1000) * 1.5 - (inlet_cod / 2000) + np.random.normal(0, 0.2, num_samples)
    effluent_do = np.clip(effluent_do, 0.5, 6.0) # mg/L
    
    # Effluent COD (how well it was treated)
    effluent_cod = inlet_cod * 0.15 # 85% removal efficiency
    effluent_cod += np.random.normal(0, 20, num_samples)
    effluent_cod = np.clip(effluent_cod, 10, 500)
    
    # Compile into DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'inlet_flow_rate_m3h': inlet_flow_rate,
        'inlet_pH': inlet_pH,
        'inlet_temp_C': inlet_temp,
        'inlet_turbidity_NTU': inlet_turbidity,
        'inlet_cod_mgL': inlet_cod,
        'chemical_dosage_Lh': chemical_dosage,
        'aeration_blower_rpm': aeration_blower_rpm,
        'effluent_pH': effluent_pH,
        'effluent_do_mgL': effluent_do,
        'effluent_cod_mgL': effluent_cod
    })
    
    # Inject missing values (NaNs) to simulate sensor drops (approx 1% data loss on some columns)
    nan_mask_flow = np.random.rand(num_samples) < 0.01
    df.loc[nan_mask_flow, 'inlet_flow_rate_m3h'] = np.nan
    
    nan_mask_cod = np.random.rand(num_samples) < 0.01
    df.loc[nan_mask_cod, 'inlet_cod_mgL'] = np.nan
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_samples} samples of synthetic WWT data at {output_path}")
    print(df.head())
    print(f"Introduced {df.isna().sum().sum()} total missing values to simulate sensor drops.")

if __name__ == "__main__":
    generate_wwt_data()
