"""
Score d'anomalie (Isolation Forest) et résistance à l'imputation
Dataset : Global Air Pollution
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ── Chargement des artefacts ──────────────────────────────────────────────────
model         = joblib.load("model.joblib")
scaler        = joblib.load("scaler.joblib")
imputer       = joblib.load("imputer.joblib")
isolation_forest = joblib.load("isolation_forest.joblib")

# ── Données ───────────────────────────────────────────────────────────────────
df = pd.read_csv("global_air_pollution_dataset.csv")

FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
TARGET   = "AQI Value"

X_raw = df[FEATURES].copy()
y     = df[TARGET].values

X_imputed = imputer.transform(X_raw)
X_scaled  = scaler.transform(X_imputed)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SCORE D'ANOMALIE — méthode du coude
# ─────────────────────────────────────────────────────────────────────────────
anomaly_scores = isolation_forest.decision_function(X_scaled)  # plus bas = plus anormal

# Courbe du coude : trier les scores et chercher le coude
sorted_scores = np.sort(anomaly_scores)

plt.figure(figsize=(10, 5))
plt.plot(sorted_scores, color="#e74c3c", linewidth=1.5)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Score d'anomalie — Isolation Forest (méthode du coude)")
plt.xlabel("Observations triées par score")
plt.ylabel("Score d'anomalie (decision_function)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("anomaly_score_curve.png", dpi=150)
plt.show()
print("Courbe d'anomalie sauvegardée : anomaly_score_curve.png")

# ── Impact du score d'anomalie sur la qualité de prédiction ──────────────────
# On découpe en déciles de score d'anomalie et on calcule la RMSE par décile
y_pred     = model.predict(X_scaled)
score_bins = pd.qcut(anomaly_scores, q=10, labels=False)

rmse_per_decile = []
for decile in range(10):
    mask = (score_bins == decile)
    rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    rmse_per_decile.append(rmse)

plt.figure(figsize=(10, 5))
bars = plt.bar(range(1, 11), rmse_per_decile,
               color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10)))
plt.title("RMSE par décile de score d'anomalie\n(décile 1 = plus anormal, 10 = plus normal)")
plt.xlabel("Décile de score d'anomalie")
plt.ylabel("RMSE")
plt.xticks(range(1, 11))
plt.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("rmse_by_anomaly_decile.png", dpi=150)
plt.show()
print("RMSE par décile sauvegardée : rmse_by_anomaly_decile.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RÉSISTANCE À L'IMPUTATION — taux de valeurs manquantes croissant
# ─────────────────────────────────────────────────────────────────────────────
y_pred_clean = model.predict(X_scaled)
rmse_clean   = np.sqrt(mean_squared_error(y, y_pred_clean))
print(f"\nRMSE baseline (sans valeurs manquantes) : {rmse_clean:.4f}")

def evaluate_imputation_impact(feature_idx: int, missing_pct: float) -> float:
    """
    Masque aléatoirement `missing_pct`% des valeurs d'une feature,
    impute, scale, prédit et retourne la variation de RMSE en %.
    """
    X_missing = X_imputed.copy().astype(float)
    n_missing  = int(len(X_missing) * missing_pct / 100)
    mask_idx   = np.random.choice(len(X_missing), size=n_missing, replace=False)
    X_missing[mask_idx, feature_idx] = np.nan

    X_reimp   = imputer.transform(X_missing)
    X_rescaled = scaler.transform(X_reimp)
    y_pred    = model.predict(X_rescaled)
    rmse      = np.sqrt(mean_squared_error(y, y_pred))
    return 100 * (rmse - rmse_clean) / rmse_clean

np.random.seed(42)
missing_rates = np.array([1, 5, 10, 20, 30, 50])

TOP3_FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value"]
top3_indices  = [FEATURES.index(f) for f in TOP3_FEATURES]

imputation_results = {}
for idx, feat in zip(top3_indices, TOP3_FEATURES):
    imputation_results[feat] = [evaluate_imputation_impact(idx, r) for r in missing_rates]

# Affichage
markers = ["o-", "s-", "^-"]
colors  = ["#e74c3c", "#3498db", "#2ecc71"]

plt.figure(figsize=(10, 6))
for (feat, vals), marker, color in zip(imputation_results.items(), markers, colors):
    plt.plot(missing_rates, vals, marker, label=feat, color=color, linewidth=2)

plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Résistance à l'Imputation — Air Pollution (AQI Value)", fontsize=14)
plt.xlabel("Taux de valeurs manquantes injectées (%)")
plt.ylabel("Variation de la RMSE (%)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("imputation_sensitivity_curve.png", dpi=150)
plt.show()
print("Courbe d'imputation sauvegardée : imputation_sensitivity_curve.png")
