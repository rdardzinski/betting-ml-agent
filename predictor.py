import os
import pickle
import numpy as np
import pandas as pd

def load_model(market="Over25"):
    filename = f"models/agent_state_{market}.pkl"
    if not os.path.exists(filename):
        # fallback do models/model_x.pkl z run_training.py
        fallback = f"models/model_{market.lower()}.pkl"
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"{filename} ani {fallback} nie istnieje. Uruchom run_training.py!")
        filename = fallback

    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    predictions = df.copy()
    markets = ["Over25","NBA"]  # multi-market: football + basketball

    # 1️⃣ Football Over25
    if "FTHG" in df.columns and "FTAG" in df.columns:
        try:
            model, acc = load_model("Over25")
            features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]
            probs_raw = model.predict_proba(features)
            probs = probs_raw[:,1] if probs_raw.shape[1]>1 else np.full((len(features),), probs_raw[0,0])
            predictions["Over25_Prob"] = probs
            predictions["Over25_Confidence"] = (probs*100).round(1)
            predictions["Over25_ValueFlag"] = probs > 0.55
            predictions["Over25_ModelAccuracy"] = acc
        except Exception as e:
            print(f"Błąd predykcji football: {e}")

    # 2️⃣ NBA HomeWin
    if "HomeScore" in df.columns and "AwayScore" in df.columns:
        try:
            model, acc = load_model("NBA")
            features = df[["HomeScore","AwayScore"]]
            probs_raw = model.predict_proba(features)
            probs = probs_raw[:,1] if probs_raw.shape[1]>1 else np.full((len(features),), probs_raw[0,0])
            predictions["HomeWin_Prob"] = probs
            predictions["HomeWin_Confidence"] = (probs*100).round(1)
            predictions["HomeWin_ValueFlag"] = probs > 0.55
            predictions["HomeWin_ModelAccuracy"] = acc
        except Exception as e:
            print(f"Błąd predykcji NBA: {e}")
            predictions["HomeWin_Prob"] = np.nan
            predictions["HomeWin_Confidence"] = np.nan
            predictions["HomeWin_ValueFlag"] = False
            predictions["HomeWin_ModelAccuracy"] = np.nan

    return predictions
