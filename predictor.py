import os
import pickle
import numpy as np
import pandas as pd

def load_model(market="Over25"):
    """
    Ładuje model i jego dokładność z pliku .pkl
    """
    filename = f"models/agent_state_{market}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!")
    
    with open(filename,"rb") as f:
        state = pickle.load(f)
    
    return state["model"], state["accuracy"]

def predict(df):
    """
    Multi-market prediction dla:
    - Piłka nożna: Over25, BTTS
    - Koszykówka: NBA HomeWin
    """
    predictions = df.copy()

    # -----------------------
    # 1️⃣ Piłka nożna
    # -----------------------
    football_markets = ["Over25","BTTS"]
    for market in football_markets:
        model, accuracy = load_model(market)
        features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]

        # bezpieczne predict_proba
        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            # tylko jedna klasa w treningu – ustaw prawdopodobieństwo 0 lub 1
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy

    # -----------------------
    # 2️⃣ Koszykówka (NBA) – HomeWin
    # -----------------------
    try:
        model, accuracy = load_model("NBA")
        features = df[["HomeScore","AwayScore"]]

        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions["HomeWin_Prob"] = probs
        predictions["HomeWin_Confidence"] = (probs*100).round(1)
        predictions["HomeWin_ValueFlag"] = probs > 0.55
        predictions["HomeWin_ModelAccuracy"] = accuracy
    except FileNotFoundError:
        # jeżeli brak modelu NBA – wypełnij NaN
        predictions["HomeWin_Prob"] = np.nan
        predictions["HomeWin_Confidence"] = np.nan
        predictions["HomeWin_ValueFlag"] = False
        predictions["HomeWin_ModelAccuracy"] = np.nan

    return predictions
