import os
import pickle
import numpy as np
import pandas as pd

# Lista obsługiwanych rynków
MARKETS = ["Over25", "BTTS", "1HGoals", "2HGoals", "Cards", "Corners"]

def load_model(market):
    """
    Wczytuje model dla danego rynku.
    Zwraca tuple (model, accuracy).
    """
    filename = f"models/model_{market}.pkl"
    if not os.path.exists(filename):
        print(f"[WARN] Model {market} not found, pomijam")
        return None, None

    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state.get("accuracy", 0.5)

def predict(df: pd.DataFrame):
    """
    Przewiduje wszystkie rynki dla podanych meczów.
    Zwraca DataFrame z dodatkowymi kolumnami:
      {market}_Prob, {market}_ValueFlag, {market}_ModelAccuracy
    """
    if df.empty:
        return df

    predictions = df.copy()

    for market in MARKETS:
        model, acc = load_model(market)
        if model is None:
            # Tworzymy placeholder
            predictions[f"{market}_Prob"] = 0.5
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = 0.5
            continue

        # Wybór kolumn feature dla modelu
        features_cols = [
            "FTHG", "FTAG", "HomeRollingGoals", "AwayRollingGoals",
            "HomeRollingConceded", "AwayRollingConceded",
            "HomeForm", "AwayForm"
        ]
        features = predictions[features_cols].copy()

        # Predykcja prawdopodobieństwa
        try:
            probs_raw = model.predict_proba(features)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(features),), probs_raw[0, 0])
            else:
                probs = probs_raw[:, 1]
        except Exception as e:
            print(f"[WARN] Prediction failed for {market}: {e}")
            probs = np.full(len(features), 0.5)

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = acc

    return predictions
