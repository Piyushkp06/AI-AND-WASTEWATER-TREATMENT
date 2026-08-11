# JSL Angul Wastewater Treatment AI/ML Project

Welcome to the **JSL Angul Wastewater Treatment AI/ML Project**. 

This repository contains the machine learning models and data processing pipelines designed to optimize the wastewater treatment operations at the JSL Angul plant. By leveraging AI, the plant can automatically adjust to incoming water quality changes, reducing chemical and energy costs while strictly adhering to State Pollution Control Board (SPCB) regulations for Zero Liquid Discharge (ZLD).

---

## 🎯 Project Objectives

1. **Chemical Dosing Optimization:** 
   - Uses an **XGBoost Regressor** to predict the optimal chemical dosage required.
   - *Goal:* Minimize chemical consumption while ensuring the effluent pH and COD remain within target parameters.
2. **Aeration & Blower Optimization:** 
   - Uses a **Random Forest Regressor** to predict Dissolved Oxygen (DO) levels and dynamically control blower RPMs.
   - *Goal:* Reduce electrical energy consumption by avoiding over-aeration during low-load periods.
3. **Continuous Runtime Monitoring:** 
   - Uses an **Isolation Forest** anomaly detection model.
   - *Goal:* Automatically flag sensor faults or toxic shock events (extreme pH drops or COD spikes) in real-time, preventing non-compliant discharge.

---

## 📂 Directory Structure

```text
JSL_Angul_WWT_Project/
├── data/
│   ├── raw/               # Contains historical/synthetic SCADA logs (e.g., synthetic_wwt_data.csv)
│   ├── processed/         # Cleaned and transformed datasets
│   └── external/          # External regulatory limits or reference data
├── notebooks/             # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                   
│   ├── data_prep/         # Scripts for data generation and preprocessing
│   ├── models/            # Model training scripts (XGBoost, Random Forest, Isolation Forest)
│   ├── control/           # Real-time control agents (Blower Control Engine)
│   └── monitoring/        # Real-time anomaly detection simulation
├── models/                # Saved serialized model artifacts (.pkl files)
├── requirements.txt       # Python dependencies required to run the project
└── README.md              # Project documentation
```

---

## 🚀 Setup & Installation

1. **Clone or Navigate to the Directory:** Ensure you are in the `JSL_Angul_WWT_Project` directory.
2. **Install Dependencies:** It is recommended to use a virtual environment. Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Usage & Model Training

To retrain the models with new historical SCADA data, place your updated `.csv` files in the `data/raw/` directory and run the following training scripts located in `src/models/`:

### 1. Data Generation (Optional)
If you need to generate a synthetic dataset for testing:
```bash
python src/data_prep/generate_synthetic_data.py
```

### 2. Data Preprocessing
After generating data, run the preprocessing script to handle missing values and engineer new features (e.g., `COD_load_kg_h`):
```bash
python src/data_prep/preprocess_data.py
```
*Outputs `processed_wwt_data.csv` to the `data/processed/` folder.*

### 3. Train Chemical Dosing Model
```bash
python src/models/train_dosing_model.py
```
*Outputs `chemical_dosing_xgb.pkl` to the `models/` folder.*

### 4. Train Aeration Control Model
```bash
python src/models/train_aeration_model.py
```
*Outputs `aeration_rf_model.pkl` to the `models/` folder.*

### 5. Train Anomaly Detection Model
```bash
python src/models/train_anomaly_detector.py
```
*Outputs `anomaly_detector_if.pkl` to the `models/` folder.*

### 6. Run Real-Time Monitor Simulation
To test the anomaly detection against simulated toxic shocks:
```bash
python src/monitoring/runtime_monitor.py
```

---

## 📈 Evaluation Results
You can find the latest training metrics, feature importance scores, and model performance details in the **[results.md](file:///c:/Users/pkpan/AI&WWT/JSL_Angul_WWT_Project/results.md)** file at the root of the project.

---

## 📊 Technologies Used
- **Python 3.x**
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-Learn** - Machine learning pipeline and models (Random Forest, Isolation Forest)
- **XGBoost** - Advanced gradient boosting regression
- **Joblib** - Model serialization

# AI-AND-WASTEWATER-TREATMENT