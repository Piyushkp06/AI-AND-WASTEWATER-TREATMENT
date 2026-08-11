import joblib
import numpy as np
import time
import os

class RuntimeMonitor:
    """
    Continuous Runtime Monitor for Regulatory Compliance.
    Uses an Isolation Forest to detect anomalous influent streams
    in real-time to alert operators before non-compliant effluent is discharged.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, '..', '..', 'models', 'anomaly_detector_if.pkl')
            
        print(f"Initializing Runtime Monitor...")
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def check_for_anomalies(self, flow, ph, temp, turbidity, cod):
        if not self.model:
            return False
            
        # Calculate engineered feature
        cod_load_kg_h = (flow * cod) / 1000.0
        
        # Features must match the training script: 
        features = np.array([[flow, ph, temp, turbidity, cod, cod_load_kg_h]])
        prediction = self.model.predict(features)[0]
        
        # Isolation Forest returns -1 for outliers, 1 for inliers
        is_anomaly = (prediction == -1)
        
        if is_anomaly:
            print(f"!!! [ALERT] ANOMALY DETECTED !!!")
            print(f"   Sensors: pH={ph}, COD={cod} mg/L, Temp={temp}C, COD Load={cod_load_kg_h:.2f} kg/h")
            print(f"   Action Required: Verify sensors or divert flow to holding tank.")
        else:
            print(f"[OK] System Normal. pH={ph}, COD={cod}, COD Load={cod_load_kg_h:.2f} kg/h")
            
        return is_anomaly

if __name__ == "__main__":
    monitor = RuntimeMonitor()
    
    print("\n--- Starting Continuous Monitoring Simulation ---")
    
    # Simulate a normal reading
    print("\n[Simulated Timestamp: 14:00]")
    monitor.check_for_anomalies(flow=330.0, ph=6.5, temp=32.0, turbidity=150.0, cod=1200.0)
    
    time.sleep(1)
    
    # Simulate a toxic shock (extreme acidic pH and high COD)
    print("\n[Simulated Timestamp: 14:15] - Toxic Shock Event")
    monitor.check_for_anomalies(flow=350.0, ph=2.1, temp=45.0, turbidity=400.0, cod=3500.0)
