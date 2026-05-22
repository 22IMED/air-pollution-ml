# 🌍 Global Air Pollution — Pipeline ML

Pipeline complet de détection d'anomalies et de prédiction d'AQI (Air Quality Index) basé sur le dataset **Global Air Pollution**.

---

## 📁 Structure du projet

```
.
├── global_air_pollution_dataset.csv   # Dataset source
├── generate_artifacts.py              # Entraîne et exporte les modèles (.joblib)
├── pipeline.py                        # Script CLI : CSV → prédiction + score d'anomalie
├── noise_sensitivity_curve.py         # Courbes de sensibilité au bruit
├── anomaly_imputation_tests.py        # Tests score d'anomalie + résistance à l'imputation
├── test_performance.py                # Tests de charge et stress (pytest)
├── model.joblib                       # Modèle principal (généré)
├── scaler.joblib                      # StandardScaler (généré)
├── imputer.joblib                     # SimpleImputer (généré)
└── isolation_forest.joblib            # Isolation Forest (généré)
```

---

## ⚙️ Installation

```bash
pip install scikit-learn pandas numpy matplotlib joblib pytest
```

---

## 🚀 Utilisation

### 1. Générer les artefacts (à faire en premier)

Entraîne les modèles et exporte les `.joblib` nécessaires à tous les autres scripts :

```bash
python generate_artifacts.py
# ou avec un chemin personnalisé :
python generate_artifacts.py --csv chemin/vers/global_air_pollution_dataset.csv
```

Produit : `imputer.joblib`, `scaler.joblib`, `isolation_forest.joblib`, `model.joblib`

---

### 2. Pipeline de prédiction

```bash
python pipeline.py --input global_air_pollution_dataset.csv
# avec export des résultats :
python pipeline.py --input global_air_pollution_dataset.csv --output predictions.csv
```

Ajoute au CSV deux colonnes : `anomaly_score` et `prediction`.

---

### 3. Courbes de sensibilité au bruit

```bash
python noise_sensitivity_curve.py
```

Génère `noise_sensitivity_curve.png` — variation de RMSE selon le bruit injecté sur les 3 features les plus corrélées (PM2.5, CO, Ozone).

---

### 4. Tests score d'anomalie + imputation

```bash
python anomaly_imputation_tests.py
```

Génère :
- `anomaly_score_curve.png` — courbe du coude des scores d'anomalie
- `rmse_by_anomaly_decile.png` — RMSE par décile de score d'anomalie
- `imputation_sensitivity_curve.png` — résistance à l'imputation selon le taux de valeurs manquantes

---

### 5. Tests de performance (charge & stress)

```bash
python -m pytest test_performance.py -v
# ou uniquement les tests marqués perf :
python -m pytest test_performance.py -m perf -v
```

4 tests inclus :

| Test | Description | Seuil |
|------|-------------|-------|
| `test_load_time_by_volume` | Montée en charge (20% → 100% du dataset) | < 10s |
| `test_memory_ceiling` | Pic mémoire dataset complet | < 500 MB |
| `test_stress_repeated_calls` | 10 exécutions consécutives | avg < 5s, std stable |
| `test_stress_concurrent_batches` | 20 batchs de tailles aléatoires | < 3s chacun |

---

## 🔬 Features utilisées

| Feature | Rôle |
|---------|------|
| `PM2.5 AQI Value` | Feature la plus corrélée avec l'AQI (r ≈ 0.98) |
| `CO AQI Value` | 2ème feature la plus corrélée |
| `Ozone AQI Value` | 3ème feature la plus corrélée |
| `NO2 AQI Value` | Feature complémentaire |
| `AQI Value` | **Target** (valeur à prédire) |

---

## 🧠 Modèles

- **RandomForestRegressor** — prédit `AQI Value` à partir des 4 features
- **IsolationForest** — détecte les anomalies (contamination = 5%)
- **SimpleImputer** (médiane) — gère les valeurs manquantes
- **StandardScaler** — normalise les features avant modélisation
