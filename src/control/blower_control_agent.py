import joblib
import numpy as np
import os

class BlowerControlAgent:
    """
    Supervisory AI Agent to optimize aeration blower runtimes.
    It uses a trained Random Forest model to predict the optimal RPM
    required to maintain a target Dissolved Oxygen (DO) level, 
    thereby minimizing unnecessary energy consumption.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, '..', '..', 'models', 'aeration_rf_model.pkl')
            
        print(f"Loading Aeration Model from {model_path}...")
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def recommend_blower_rpm(self, inlet_flow, inlet_cod, target_do=2.0):
        """
        Recommends blower RPM.
        Default target_do is 2.0 mg/L which is typical for standard biological treatment.
        """
        if not self.model:
            return 1500.0 # Safe default fallback
        
        # Calculate engineered feature
        cod_load_kg_h = (inlet_flow * inlet_cod) / 1000.0
        
        # Features must match the training script: 
        # ['inlet_flow_rate_m3h', 'inlet_cod_mgL', 'COD_load_kg_h', 'effluent_do_mgL']
        features = np.array([[inlet_flow, inlet_cod, cod_load_kg_h, target_do]])
        recommended_rpm = self.model.predict(features)[0]
        
        # Enforce physical hardware limits
        recommended_rpm = np.clip(recommended_rpm, 500, 3000)
        
        return recommended_rpm

if __name__ == "__main__":
    agent = BlowerControlAgent()
    # Simulated current plant conditions
    current_flow = 350.0  # m3/h
    current_cod = 1400.0  # mg/L
    desired_do = 2.5      # mg/L DO target
    
    rpm = agent.recommend_blower_rpm(current_flow, current_cod, desired_do)
    print(f"\n--- AI Control Recommendation ---")
    print(f"Current Plant Load: Flow={current_flow} m3/h, COD={current_cod} mg/L")
    print(f"Target DO: {desired_do} mg/L")
    print(f"Recommended Blower Speed: {rpm:.0f} RPM")
