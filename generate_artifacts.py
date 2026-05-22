"""
generate_artifacts.py
=====================
Génère les 4 artefacts .joblib requis par les scripts existants :
  - imputer.joblib
  - scaler.joblib
  - model.joblib          (RandomForestRegressor — prédit AQI Value)
  - isolation_forest.joblib

Usage :
    python generate_artifacts.py
    python generate_artifacts.py --csv chemin/vers/global_air_pollution_dataset.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
TARGET   = "AQI Value"

def main(csv_path: str) -> None:
    print(f"[1/5] Chargement de {csv_path} …")
    df = pd.read_csv(csv_path)
    X  = df[FEATURES].copy()
    y  = df[TARGET].values
    print(f"      {len(df)} lignes chargées")

    # ── Imputer ───────────────────────────────────────────────────────────────
    print("[2/5] Entraînement de l'imputer (median) …")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    joblib.dump(imputer, "imputer.joblib")
    print("      → imputer.joblib sauvegardé")

    # ── Scaler ────────────────────────────────────────────────────────────────
    print("[3/5] Entraînement du scaler (StandardScaler) …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    joblib.dump(scaler, "scaler.joblib")
    print("      → scaler.joblib sauvegardé")

    # ── Isolation Forest ──────────────────────────────────────────────────────
    print("[4/5] Entraînement de l'Isolation Forest …")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    joblib.dump(iso, "isolation_forest.joblib")
    print("      → isolation_forest.joblib sauvegardé")

    # ── Modèle principal (RandomForest Regressor) ─────────────────────────────
    print("[5/5] Entraînement du modèle principal (RandomForestRegressor) …")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    joblib.dump(model, "model_2.joblib'.joblib")
    print("      → model.joblib sauvegardé")

    print("\n✅ Les 4 artefacts sont prêts :")
    for f in ["imputer.joblib", "scaler.joblib", "isolation_forest.joblib", "model.joblib"]:
        print(f"   {f}")
    print("\nTu peux maintenant lancer tes scripts :")
    print("   python noise_sensitivity_curve.py")
    print("   python anomaly_imputation_tests.py")
    print("   python pipeline.py --input global_air_pollution_dataset.csv --output predictions.csv")
    print("   pytest test_performance.py -m perf -v")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="global_air_pollution_dataset.csv",
        help="Chemin vers le CSV de pollution (défaut : global_air_pollution_dataset.csv)"
    )
    args = parser.parse_args()
    main(args.csv)
