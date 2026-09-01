"""
Centrifugal Pump Digital Twin — FIXED ML Training Pipeline
Saves with keys: 'rul_model', 'health_model', 'anomaly_model', 'scaler', 'metadata'
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==================== CONFIGURATION ====================
CSV_PATH = "converted_file.csv"  # <-- Use your actual CSV filename (no spaces = fewer errors)
MODEL_OUTPUT = "pump_model.pkl"

FEATURE_COLS = [
    'A_ACR_Mot.PV', 'A_ACR_Mot.SV', 'A_ACR_Mot.TV',
    'A_ACR_Pmp.PV', 'A_ACR_Pmp.SV', 'A_ACR_Pmp.TV',
    'A_Pres.PV', 'A_Temp.PV', 'Vibration_hz',
    'Temperature', 'Load_Percent', 'RPM', 'Barometer'
]

TARGET_RUL = 'RUL_hours'

# ==================== FIND CSV ====================
print("Looking for CSV file...")

search_paths = [
    CSV_PATH,
    f"../{CSV_PATH}",
    "../ml/converted_file (1).csv",
    "converted_file (1).csv",
    "../converted_file (1).csv",
    "data.csv",
    "../data.csv",
]

df = None
for p in search_paths:
    if Path(p).exists():
        print(f"Found: {p}")
        df = pd.read_csv(p)
        break

if df is None:
    print("CSV not found! Searched:")
    for p in search_paths:
        print(f"  - {p}")
    print("\nPlease place your CSV in the ml/ folder and update CSV_PATH.")
    exit(1)

print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# ==================== DATA PREP ====================
available_features = [c for c in FEATURE_COLS if c in df.columns]
missing = [c for c in FEATURE_COLS if c not in df.columns]

if missing:
    print(f"Missing features (ignored): {missing}")
    print(f"Using available: {available_features}")
else:
    print("All 13 features found!")

if TARGET_RUL not in df.columns:
    print(f"RUL column '{TARGET_RUL}' not found!")
    candidates = [c for c in df.columns if 'rul' in c.lower() or 'hour' in c.lower()]
    if candidates:
        print(f"Did you mean: {candidates}?")
    exit(1)

df[available_features] = df[available_features].ffill().bfill()
df = df.dropna(subset=[TARGET_RUL])

max_rul = df[TARGET_RUL].max()
df['health_score'] = (df[TARGET_RUL] / max_rul * 100).clip(0, 100)

vib_thresh = df['Vibration_hz'].quantile(0.9) if 'Vibration_hz' in df.columns else 0
rul_thresh = df[TARGET_RUL].quantile(0.1)
pres_thresh = df['A_Pres.PV'].quantile(0.05) if 'A_Pres.PV' in df.columns else 0

anomaly_conditions = df[TARGET_RUL] < rul_thresh
if 'Vibration_hz' in df.columns:
    anomaly_conditions = anomaly_conditions | (df['Vibration_hz'] > vib_thresh)
if 'A_Pres.PV' in df.columns:
    anomaly_conditions = anomaly_conditions | (df['A_Pres.PV'] < pres_thresh)

df['anomaly'] = anomaly_conditions.astype(int)

def get_status(row):
    if row['health_score'] >= 80 and row['anomaly'] == 0:
        return 'Healthy'
    elif row['health_score'] >= 50:
        return 'Warning'
    else:
        return 'Critical'

df['health_status'] = df.apply(get_status, axis=1)

print(f"\nData Summary:")
print(f"  Healthy: {(df['health_status']=='Healthy').sum()}")
print(f"  Warning: {(df['health_status']=='Warning').sum()}")
print(f"  Critical: {(df['health_status']=='Critical').sum()}")

# ==================== MODEL TRAINING ====================
X = df[available_features].values
y_rul = df[TARGET_RUL].values
y_health = df['health_score'].values
y_anomaly = df['anomaly'].values

X_train, X_test, yr_train, yr_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
    X, y_rul, y_health, y_anomaly, test_size=0.2, random_state=42
)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nTraining RUL Regressor...")
rul_model = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
rul_model.fit(X_train_s, yr_train)

print("Training Health Score Regressor...")
health_model = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=3, random_state=42, n_jobs=-1)
health_model.fit(X_train_s, yh_train)

print("Training Anomaly Classifier...")
anomaly_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=3, random_state=42, n_jobs=-1)
anomaly_model.fit(X_train_s, ya_train)

# ==================== EVALUATION ====================
rul_pred = rul_model.predict(X_test_s)
health_pred = health_model.predict(X_test_s)
anomaly_pred = anomaly_model.predict(X_test_s)

metadata = {
    'feature_names': available_features,
    'feature_importance': {name: float(imp) for name, imp in zip(available_features, health_model.feature_importances_)},
    'rul_mae': float(mean_absolute_error(yr_test, rul_pred)),
    'rul_r2': float(r2_score(yr_test, rul_pred)),
    'health_r2': float(r2_score(yh_test, health_pred)),
    'anomaly_acc': float((ya_test == anomaly_pred).mean()),
    'max_rul': float(max_rul),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

print(f"\nMODEL PERFORMANCE:")
print(f"  RUL MAE: {metadata['rul_mae']:.2f} hours")
print(f"  RUL R2:  {metadata['rul_r2']:.4f}")
print(f"  Health R2: {metadata['health_r2']:.4f}")
print(f"  Anomaly Accuracy: {metadata['anomaly_acc']:.4f}")

print(f"\nTop 5 Important Features:")
for name, imp in sorted(metadata['feature_importance'].items(), key=lambda x: -x[1])[:5]:
    print(f"  {name:20s}: {imp:.4f}")

# ==================== SAVE ARTIFACTS ====================
# CRITICAL: These keys MUST match what model.py expects!
artifacts = {
    'rul_model': rul_model,
    'health_model': health_model,
    'anomaly_model': anomaly_model,
    'scaler': scaler,
    'feature_names': available_features,
    'metadata': metadata
}

joblib.dump(artifacts, MODEL_OUTPUT, compress=3)
print(f"\nModel saved to: {MODEL_OUTPUT}")
print(f"  File size: {Path(MODEL_OUTPUT).stat().st_size / 1024:.1f} KB")
print(f"  Keys saved: {list(artifacts.keys())}")
print("\nTraining complete! Copy pump_model.pkl to backend/ folder.")
