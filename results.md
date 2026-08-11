# Project Evaluation Results

This file stores the latest training and evaluation results for the machine learning models developed in the JSL Angul Wastewater Treatment Project.

## 1. Aeration Control Model (Random Forest Regressor)
**Goal:** Recommend optimal Blower RPM based on inlet flow, COD, and target DO.

*   **Training Samples:** 8000
*   **RMSE:** 48.46 RPM
*   **R² Score:** 0.9262
*   **Model Saved At:** `models/aeration_rf_model.pkl`

## 2. Anomaly Detection Model (Isolation Forest)
**Goal:** Detect toxic shocks and abnormal events from influent parameters.

*   **Training Samples:** 10,000 historical samples
*   **Contamination Factor:** 0.01 (1%)
*   **Anomalies Detected in Training:** 100 (1.00%)
*   **Model Saved At:** `models/anomaly_detector_if.pkl`

## 3. Chemical Dosing Model (XGBoost Regressor)
**Goal:** Predict optimal chemical dosage required for neutralization.

*   **Training Samples:** 8000
*   **RMSE:** 2.11 L/h
*   **MAE:** 1.70 L/h
*   **R² Score:** 0.9834
*   **Model Saved At:** `models/chemical_dosing_xgb.pkl`

### Feature Importances (Dosing Model)
*   **inlet_pH:** 0.9940
*   **COD_load_kg_h:** 0.0013
*   **inlet_flow_rate_m3h:** 0.0012
*   **inlet_temp_C:** 0.0012
*   **inlet_cod_mgL:** 0.0012
*   **inlet_turbidity_NTU:** 0.0011

---
*Results generated on: 2026-08-11*
