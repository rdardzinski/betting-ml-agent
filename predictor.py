import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "models"

def _ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def load_model(market="Over25"):
    _ensure_model_dir()
    filename = os.path.join(MODEL_DIR,f"agent_state_{market}.pkl")
    if not os.path.exists(filename):
        # fallback: utwórz dummy model
        print(f"[WARN] {filename} nie istnieje, tworzony dummy model.")
        model = RandomForestClassifier()
        model.fit(np.zeros((1,5)), [0])
        return model, 0.5
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    predictions = df.copy()
    football_markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    basketball_markets = ["HomeWin","BasketPoints","BasketSum"]

    # --- FOOTBALL ---
    if "FTHG" in df.columns and "FTAG" in df.columns:
        features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals","1HGoals","2HGoals","BTTS","Cards","Corners"]].copy()
        for market in football_markets:
            model, acc = load_model(market)
            try:
                probs_raw = model.predict_proba(features)
                if probs_raw.shape[1]==1:
                    probs = np.full(len(features),probs_raw[0,0])
                else:
                    probs = probs_raw[:,1]
            except Exception:
                probs = np.full(len(features),0.5)
            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs>0.55
            predictions[f"{market}_ModelAccuracy"] = acc

    # --- BASKETBALL ---
    if "HomeScore" in df.columns and "AwayScore" in df.columns:
        features_b = df[["HomeScore","AwayScore"]].copy()
        for market in basketball_markets:
            model, acc = load_model(market)
            try:
                probs_raw = model.predict_proba(np.ones((len(features_b),2)))
                if probs_raw.shape[1]==1:
                    probs = np.full(len(features_b),probs_raw[0,0])
                else:
                    probs = probs_raw[:,1]
            except Exception:
                probs = np.full(len(features_b),0.55)
            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs>0.55
            predictions[f"{market}_ModelAccuracy"] = acc

    return predictions
