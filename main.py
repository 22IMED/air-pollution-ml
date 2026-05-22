import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────────────────────
# 1. Préparation des données
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('global_air_pollution_dataset.csv')

FEATURES = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
TARGET   = "AQI Value"

X = df[FEATURES].copy()
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Imputation et Scaling (fit sur train uniquement)
# ─────────────────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled  = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled  = scaler.transform(X_test_imputed)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Entraînement du modèle et Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_scaled)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Méthode du coude (RMSE vs Coverage)
# ─────────────────────────────────────────────────────────────────────────────
anomaly_scores = iso_forest.score_samples(X_test_scaled)
y_pred         = model.predict(X_test_scaled)
rmse_base      = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE baseline : {rmse_base:.4f}")

thresholds = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
count, rmse_list = [], []

for t in thresholds:
    mask = anomaly_scores >= t
    count.append(100 * mask.sum() / len(y_test))
    if mask.sum() > 0:
        error = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
        rmse_list.append(error)
    else:
        rmse_list.append(np.nan)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds, rmse_list, label='RMSE', color='blue')
ax1.axvline(x=-0.5, color='green', linestyle='--', label='Seuil suggéré')
ax1.set_ylabel('RMSE')
ax1.set_xlabel('Seuil de score d\'anomalie')
ax2 = ax1.twinx()
ax2.plot(thresholds, count, label='Coverage %', color='red', linestyle='--')
ax2.set_ylabel('Coverage (%)')
ax1.set_title('Méthode du coude : RMSE vs Coverage — Air Pollution')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('anomaly_elbow_curve.png', dpi=150)
plt.show()
print("Courbe du coude sauvegardée : anomaly_elbow_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Courbes de sensibilité au bruit (3 features les plus corrélées)
# ─────────────────────────────────────────────────────────────────────────────
TOP3 = ["PM2.5 AQI Value", "CO AQI Value", "Ozone AQI Value"]
noise_levels = np.array([1, 3, 5, 10, 15, 20, 30, 50])

np.random.seed(42)
plt.figure(figsize=(10, 6))
markers = ["o-", "s-", "^-"]
colors  = ["#e74c3c", "#3498db", "#2ecc71"]

for feat, marker, color in zip(TOP3, markers, colors):
    feat_idx = FEATURES.index(feat)
    rmse_noise = []
    for pct in noise_levels:
        X_noisy = X_test_imputed.copy()
        std_dev  = X_noisy[:, feat_idx].std()
        noise    = np.random.normal(0, std_dev * pct / 100, size=X_noisy.shape[0])
        X_noisy[:, feat_idx] += noise
        X_noisy_scaled = scaler.transform(X_noisy)
        preds = model.predict(X_noisy_scaled)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        rmse_noise.append(100 * (rmse - rmse_base) / rmse_base)
    plt.plot(noise_levels, rmse_noise, marker, label=feat, color=color, linewidth=2)

plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Sensibilité au Bruit — Air Pollution (AQI Value)', fontsize=14)
plt.xlabel("Niveau de bruit (% de l'écart-type)")
plt.ylabel('Variation de la RMSE (%)')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('noise_sensitivity_curve.png', dpi=150)
plt.show()
print("Courbe de bruit sauvegardée : noise_sensitivity_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Robustesse à l'imputation — pour chaque feature
# ─────────────────────────────────────────────────────────────────────────────
n_missing_range = range(10, 150, 20)

for idx, col_name in enumerate(FEATURES):
    rmse_per_feature = []
    for n_missing in n_missing_range:
        temp_rmse = []
        for _ in range(5):  # Moyenne sur 5 tirages
            X_corrupted = X_test.copy()
            indices = np.random.choice(len(X_corrupted), n_missing, replace=False)
            X_corrupted.iloc[indices, idx] = np.nan

            X_imp = imputer.transform(X_corrupted)
            X_sc  = scaler.transform(X_imp)
            preds = model.predict(X_sc)
            temp_rmse.append(np.sqrt(mean_squared_error(y_test, preds)))
        rmse_per_feature.append(np.mean(temp_rmse))

    plt.figure()
    plt.plot(n_missing_range, rmse_per_feature, marker='o', color='purple')
    plt.title(f'Robustesse Imputation : {col_name}')
    plt.xlabel('Nombre de valeurs manquantes')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'imputation_{col_name.replace(" ", "_")}.png', dpi=150)
    plt.show()
    print(f"Courbe imputation sauvegardée : imputation_{col_name.replace(' ', '_')}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Sauvegarde des artefacts
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(model,      'model.joblib')
joblib.dump(iso_forest, 'isolation_forest.joblib')
joblib.dump(scaler,     'scaler.joblib')
joblib.dump(imputer,    'imputer.joblib')
print("\nModèles et outils sauvegardés !")
