import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =========================
# ŁADOWANIE MODELÓW
# =========================
def load_model(market="Over25"):
    filename = f"models/agent_state_{market}.pkl"
    if not os.path.exists(filename):
        print(f"[WARN] Model {market} not found, pomijam")
        return None, None

    with open(filename, "rb") as f:
        state = pickle.load(f)

    return state.get("model"), state.get("accuracy", 0.0)

# =========================
# PREDYKCJE
# =========================
def predict(df):
    """
    Przyjmuje dataframe meczów piłki nożnej i zwraca dataframe z kolumnami:
    Over25_Prob, BTTS_Prob, 1HGoals_Prob, 2HGoals_Prob, Cards_Prob, Corners_Prob
    """

    if df.empty:
        return df, []

    predictions = df.copy()
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    available_markets = []

    for market in markets:
        model, accuracy = load_model(market)
        if model is None:
            continue

        # przygotowanie features – jeśli brak kolumn, uzupełnij zerami
        features_cols = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]
        for col in features_cols:
            if col not in predictions.columns:
                predictions[col] = 0

        features = predictions[features_cols]

        # bezpieczne predict_proba
        try:
            probs_raw = model.predict_proba(features)
            if probs_raw.shape[1] == 1:
                # tylko jedna klasa w treningu
                probs = np.full((len(features),), probs_raw[0,0])
            else:
                probs = probs_raw[:,1]

        except Exception as e:
            print(f"[ERROR] Prediction failed for {market}: {e}")
            probs = np.zeros(len(features))

        # kolumny wynikowe
        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy

        available_markets.append(market)

    return predictions, available_markets
