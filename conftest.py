import pytest
import joblib
import pandas as pd

@pytest.fixture(scope="session")
def trained_model():
    # Vous changerez ce fichier par "model_2.joblib" lors de l'étape de comparaison
    return joblib.load('model_2.joblib')

@pytest.fixture(scope="session")
def isolation_forest():
    return joblib.load('isolation_forest.joblib')

@pytest.fixture(scope="session")
def clean_data():
    df = pd.read_csv("global_air_pollution_dataset.csv")
    # Simulation de nettoyage simple pour l'exemple
    df = df.dropna(subset=['AQI Value', 'PM2.5 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value'])
    X = df[['PM2.5 AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value']].values
    y = df['AQI Value'].values
    return X, y, df