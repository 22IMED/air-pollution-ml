import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1. Chargement des données (adaptez le chemin si besoin)
df = pd.read_csv("global_air_pollution_dataset.csv")
df = df.dropna(subset=['AQI Value', 'PM2.5 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value'])

X = df[['PM2.5 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value']].values
y = df['AQI Value'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entraînement du NOUVEAU modèle (pas de tuning, paramètres par défaut)
print("Entraînement du GradientBoostingRegressor...")
model_2 = GradientBoostingRegressor(random_state=42)
model_2.fit(X_train, y_train)

# 3. Exportation avec joblib
joblib.dump(model_2, 'model_2.joblib')
print("Nouveau modèle exporté sous 'model_2.joblib'")