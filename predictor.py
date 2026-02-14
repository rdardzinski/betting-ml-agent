import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_model(market="Over25"):
    """
    Wczytuje model z pliku pickle.
    """
    filename = f"agent_state_{market}.pkl"
    try:
        with open(filename,"rb") as f:
            state = pickle.load(f)
        return state["model"], state["accuracy"]
    except FileNotFoundError:
        print(f"[WARN] {filename} nie istnieje. Model będzie tworzony na nowo przy nauce.")
        return None, 0.5

def predict(df):
    """
    Multi-market prediction dla football i basketball
    """
    predictions = df.copy()

    football_markets = ["Over25","BTTS","HomeGoal","AwayGoal","TotalGoals"]
    basketball_markets = ["HomeWin","HomeScore","AwayScore","TotalPoints"]

    # --- FOOTBALL ---
    for market in football_markets:
        model, acc = load_model(market)
        if model is None:
            predictions[f"{market}_Prob"] = 0.5
            predictions[f"{market}_Confidence"] = 50.0
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = acc
            continue

        features = df[["FTHG","FTAG"]].fillna(0)
        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = acc

    # --- BASKETBALL ---
    for market in basketball_markets:
        model, acc = load_model(market)
        if model is None:
            predictions[f"{market}_Prob"] = 0.55 if market=="HomeWin" else 0.5
            predictions[f"{market}_Confidence"] = 50.0
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = acc
            continue

        features = df[["HomeScore","AwayScore"]].fillna(0)
        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = acc

    return predictions
