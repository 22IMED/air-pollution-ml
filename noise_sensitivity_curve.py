"""
Courbe de sensibilité au bruit — Dataset Global Air Pollution
Features testées : PM2.5 AQI Value, CO AQI Value, Ozone AQI Value
(les 3 plus corrélées avec AQI Value, la target)
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ── Chargement des artefacts ──────────────────────────────────────────────────
model  = joblib.load("model_2.joblib")
scaler = joblib.load("scaler.joblib")
imputer = joblib.load("imputer.joblib")

# ── Données ───────────────────────────────────────────────────────────────────
df = pd.read_csv("global_air_pollution_dataset.csv")

FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
TARGET   = "AQI Value"

X_raw = df[FEATURES].copy()
y     = df[TARGET].values

# Imputation puis scaling (état « propre »)
X_imputed = imputer.transform(X_raw)
X_scaled  = scaler.transform(X_imputed)

# RMSE de référence (sans bruit)
y_pred_clean = model.predict(X_scaled)
rmse_clean   = np.sqrt(mean_squared_error(y, y_pred_clean))
print(f"RMSE de référence : {rmse_clean:.4f}")

# ── Fonction d'évaluation ─────────────────────────────────────────────────────
def evaluate_noise_impact(feature_idx: int, noise_pct: float) -> float:
    """
    Ajoute un bruit gaussien sur une seule feature (avant scaling),
    retourne la variation de RMSE en % par rapport à la baseline.
    """
    X_noisy = X_imputed.copy()
    std_dev  = X_noisy[:, feature_idx].std()
    noise    = np.random.normal(loc=0,
                                scale=std_dev * noise_pct / 100,
                                size=X_noisy.shape[0])
    X_noisy[:, feature_idx] += noise
    X_noisy_scaled = scaler.transform(X_noisy)

    y_pred_noisy = model.predict(X_noisy_scaled)
    rmse_noisy   = np.sqrt(mean_squared_error(y, y_pred_noisy))
    return 100 * (rmse_noisy - rmse_clean) / rmse_clean

# ── Calcul des courbes ────────────────────────────────────────────────────────
np.random.seed(42)
intensities = np.array([1, 3, 5, 10, 15, 20, 30, 50])

# 3 features les plus corrélées
TOP3_FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value"]
top3_indices  = [FEATURES.index(f) for f in TOP3_FEATURES]

results = {}
for idx, feat in zip(top3_indices, TOP3_FEATURES):
    results[feat] = [evaluate_noise_impact(idx, n) for n in intensities]

# ── Affichage ─────────────────────────────────────────────────────────────────
markers = ["o-", "s-", "^-"]
colors  = ["#e74c3c", "#3498db", "#2ecc71"]

plt.figure(figsize=(10, 6))
for (feat, vals), marker, color in zip(results.items(), markers, colors):
    plt.plot(intensities, vals, marker, label=feat, color=color, linewidth=2)

plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Test de Robustesse au Bruit — Air Pollution (AQI Value)", fontsize=14)
plt.xlabel("Niveau de bruit (% de l'écart-type de la feature)")
plt.ylabel("Variation de la RMSE (%)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("noise_sensitivity_curve_model_2.png", dpi=150)
plt.show()
print("Courbe sauvegardée : noise_sensitivity_curve.png")
