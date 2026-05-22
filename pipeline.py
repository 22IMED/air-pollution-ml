"""
Script global de prédiction — Air Pollution AQI
Usage : python pipeline.py --input data.csv [--output predictions.csv]

Étapes :
    1. Chargement du CSV
    2. Imputation des valeurs manquantes
    3. Normalisation (StandardScaler)
    4. Score d'anomalie (Isolation Forest)
    5. Prédiction (modèle principal)
    6. Export des résultats
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import time

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURES        = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
TARGET          = "AQI Value"
MODEL_PATH      = "model.joblib"
SCALER_PATH     = "scaler.joblib"
IMPUTER_PATH    = "imputer.joblib"
ISO_FOREST_PATH = "isolation_forest.joblib"

# ── Chargement des artefacts (fait une seule fois au démarrage) ───────────────
def load_artifacts():
    model            = joblib.load(MODEL_PATH)
    scaler           = joblib.load(SCALER_PATH)
    imputer          = joblib.load(IMPUTER_PATH)
    isolation_forest = joblib.load(ISO_FOREST_PATH)
    return model, scaler, imputer, isolation_forest


def run_pipeline(csv_path: str,
                 model, scaler, imputer, isolation_forest,
                 output_path: str | None = None) -> pd.DataFrame:
    """
    Exécute le pipeline complet sur un fichier CSV.
    Retourne un DataFrame avec les colonnes originales + anomaly_score + prediction.
    """
    t0 = time.perf_counter()

    # 1. Chargement
    df = pd.read_csv(csv_path)
    X_raw = df[FEATURES].copy()

    # 2. Imputation
    X_imputed = imputer.transform(X_raw)

    # 3. Scaling
    X_scaled = scaler.transform(X_imputed)

    # 4. Score d'anomalie
    anomaly_scores = isolation_forest.decision_function(X_scaled)   # score continu
    anomaly_labels = isolation_forest.predict(X_scaled)              # 1=normal, -1=anomalie

    # 5. Prédiction
    predictions = model.predict(X_scaled)

    # 6. Construction du résultat
    df_out = df.copy()
    df_out["anomaly_score"] = anomaly_scores
    df_out["is_anomaly"]    = (anomaly_labels == -1).astype(int)
    df_out["prediction"]    = predictions

    elapsed = time.perf_counter() - t0
    print(f"Pipeline exécuté en {elapsed:.4f}s sur {len(df)} lignes")
    print(f"  Anomalies détectées : {df_out['is_anomaly'].sum()} "
          f"({100*df_out['is_anomaly'].mean():.1f}%)")

    if output_path:
        df_out.to_csv(output_path, index=False)
        print(f"  Résultats sauvegardés : {output_path}")

    return df_out


# ── Entrypoint CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Air Pollution AQI")
    parser.add_argument("--input",  required=True, help="Chemin vers le CSV d'entrée")
    parser.add_argument("--output", default=None,  help="Chemin vers le CSV de sortie (optionnel)")
    args = parser.parse_args()

    model, scaler, imputer, isolation_forest = load_artifacts()
    results = run_pipeline(
        csv_path=args.input,
        model=model, scaler=scaler,
        imputer=imputer, isolation_forest=isolation_forest,
        output_path=args.output,
    )
    print(results[["anomaly_score", "is_anomaly", "prediction"]].describe())
