"""
Centrifugal Pump Digital Twin — ML Training Pipeline
Trains 3 models on REAL sensor data (13 features):
  1. RUL Regressor      → Predicts Remaining Useful Life
  2. Health Regressor   → Predicts Health Score (0-100)
  3. Anomaly Classifier → Detects abnormal operation

Input:  converted_file (1).csv  (your real pump data)
Output: ml/pump_model.pkl       (pre-trained artifact for backend)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==================== CONFIGURATION ====================
CSV_PATH = "converted_file (1).csv"  # <-- YOUR REAL DATA
MODEL_OUTPUT = "pump_model.pkl"

# 13 Real Sensor Features from your CSV
FEATURE_COLS = [
    'A_ACR_Mot.PV',    # Motor Actual Current
    'A_ACR_Mot.SV',    # Motor Setpoint
    'A_ACR_Mot.TV',    # Motor Totalizer
    'A_ACR_Pmp.PV',    # Pump Actual Current
    'A_ACR_Pmp.SV',    # Pump Setpoint
    'A_ACR_Pmp.TV',    # Pump Totalizer
    'A_Pres.PV',       # Pressure
    'A_Temp.PV',       # Temperature
    'Vibration_hz',    # Vibration Frequency
    'Temperature',     # Ambient/Oil Temperature
    'Load_Percent',    # Load %
    'RPM',             # Pump Speed
    'Barometer',       # Barometric Pressure
]

TARGET_RUL = 'RUL_hours'

# ==================== DATA LOADING ====================
print("🔧 Loading real pump data...")
df = pd.read_csv(CSV_PATH)

# Clean: forward-fill then backward-fill missing sensor values
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill')
df = df.dropna(subset=[TARGET_RUL])

# Derive Health Score from RUL (0-100 scale)
max_rul = df[TARGET_RUL].max()
df['health_score'] = (df[TARGET_RUL] / max_rul * 100).clip(0, 100)

# Derive Anomaly Labels (top 10% vibration OR bottom 10% RUL)
vib_thresh = df['Vibration_hz'].quantile(0.9)
rul_thresh = df[TARGET_RUL].quantile(0.1)
pres_thresh = df['A_Pres.PV'].quantile(0.05)

df['anomaly'] = (
    (df['Vibration_hz'] > vib_thresh) |
    (df[TARGET_RUL] < rul_thresh) |
    (df['A_Pres.PV'] < pres_thresh)
).astype(int)

# Derive Health Status
def get_status(row):
    if row['health_score'] >= 80 and row['anomaly'] == 0:
        return 'Healthy'
    elif row['health_score'] >= 50:
        return 'Warning'
    else:
        return 'Critical'

df['health_status'] = df.apply(get_status, axis=1)

print(f"✅ Loaded {len(df)} samples")
print(f"   Health: {(df['health_status']=='Healthy').sum()} | Warning: {(df['health_status']=='Warning').sum()} | Critical: {(df['health_status']=='Critical').sum()}")

# ==================== MODEL TRAINING ====================
X = df[FEATURE_COLS].values
y_rul = df[TARGET_RUL].values
y_health = df['health_score'].values
y_anomaly = df['anomaly'].values

X_train, X_test, yr_train, yr_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
    X, y_rul, y_health, y_anomaly, test_size=0.2, random_state=42
)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n🌲 Training RUL Regressor...")
rul_model = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
rul_model.fit(X_train_s, yr_train)

print("🌲 Training Health Score Regressor...")
health_model = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
health_model.fit(X_train_s, yh_train)

print("🌲 Training Anomaly Classifier...")
anomaly_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=3, random_state=42, n_jobs=-1)
anomaly_model.fit(X_train_s, ya_train)

# ==================== EVALUATION ====================
rul_pred = rul_model.predict(X_test_s)
health_pred = health_model.predict(X_test_s)
anomaly_pred = anomaly_model.predict(X_test_s)

metadata = {
    'feature_names': FEATURE_COLS,
    'feature_importance': {name: float(imp) for name, imp in zip(FEATURE_COLS, health_model.feature_importances_)},
    'rul_mae': float(mean_absolute_error(yr_test, rul_pred)),
    'rul_r2': float(r2_score(yr_test, rul_pred)),
    'health_r2': float(r2_score(yh_test, health_pred)),
    'anomaly_acc': float((ya_test == anomaly_pred).mean()),
    'max_rul': float(max_rul),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

print(f"\n📊 MODEL PERFORMANCE:")
print(f"   RUL MAE: {metadata['rul_mae']:.2f} hours")
print(f"   RUL R²:  {metadata['rul_r2']:.4f}")
print(f"   Health R²: {metadata['health_r2']:.4f}")
print(f"   Anomaly Accuracy: {metadata['anomaly_acc']:.4f}")

print(f"\n🔍 Top 5 Important Features:")
for name, imp in sorted(metadata['feature_importance'].items(), key=lambda x: -x[1])[:5]:
    print(f"   {name:20s}: {imp:.4f}")

# ==================== SAVE ARTIFACTS ====================
artifacts = {
    'rul_model': rul_model,
    'health_model': health_model,
    'anomaly_model': anomaly_model,
    'scaler': scaler,
    'feature_names': FEATURE_COLS,
    'metadata': metadata
}

joblib.dump(artifacts, MODEL_OUTPUT, compress=3)
print(f"\n💾 Model saved to: {MODEL_OUTPUT}")
print(f"   File size: {Path(MODEL_OUTPUT).stat().st_size / 1024:.1f} KB")
print("\n✅ Training complete! Copy pump_model.pkl to backend/ folder.")