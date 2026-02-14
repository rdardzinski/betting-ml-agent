import os
import json
import pandas as pd
from datetime import datetime
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
import pickle
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# =========================
# Funkcja wczytująca model
# =========================
def load_model(market):
    fname = os.path.join(MODEL_DIR, f"agent_state_{market}.pkl")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} not found. Run training first!")
    with open(fname, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

# =========================
# Predykcja
# =========================
def predict(df):
    predictions = df.copy()
    
    football_markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    basketball_markets = ["HomeWin","BasketPoints","BasketSum"]
    
    # --- Football ---
    for market in football_markets:
        try:
            model, acc = load_model(market)
            features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals","1HGoals","2HGoals","BTTS","Cards","Corners"]
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            X = df[features]
            probs_raw = model.predict_proba(X)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(X),), probs_raw[0,0])
            else:
                probs = probs_raw[:,1]
            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = acc
        except Exception as e:
            print(f"[ERROR] Football prediction {market}: {e}")
    
    # --- Basketball ---
    for market in basketball_markets:
        try:
            model, acc = load_model(market)
            features = ["HomeScore","AwayScore"]
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            X = df[features]
            probs_raw = model.predict_proba(X)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(X),), probs_raw[0,0])
            else:
                probs = probs_raw[:,1]
            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = acc
        except Exception as e:
            print(f"[ERROR] Basketball prediction {market}: {e}")
    
    return predictions

# =========================
# Generowanie kuponów
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("Over25_Prob", ascending=False)  # przykładowo sortujemy po football Over25
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])
    return coupons

# =========================
# Główna funkcja agenta
# =========================
def run():
    # --- Football ---
    football = get_next_matches()
    if not football.empty and "Date" in football.columns:
        football = build_features(football)
        football["Sport"] = "Football"
        football["ValueScore"] = 0.0
        football = football.reset_index(drop=True)
    else:
        football = pd.DataFrame()
        print("[WARN] No football matches")

    # --- Basketball ---
    basketball = get_basketball_games()
    if not basketball.empty and "Date" in basketball.columns:
        basketball["Sport"] = "Basketball"
        basketball["HomeWin_Prob"] = 0.55
        basketball["BasketPoints_Prob"] = 0.55
        basketball["BasketSum_Prob"] = 0.55
        basketball["ValueScore"] = 0.55
        basketball = basketball.reset_index(drop=True)
    else:
        basketball = pd.DataFrame()
        print("[WARN] No basketball matches")

    # --- Łączenie ---
    combined = pd.concat([football, basketball], ignore_index=True, sort=False)

    # --- Predykcje ---
    combined_pred = predict(combined)

    # --- Top 30% ---
    threshold = combined_pred["ValueScore"].quantile(0.7) if "ValueScore" in combined_pred.columns else 0.5
    top_pred = combined_pred[combined_pred["ValueScore"] >= threshold]

    # --- Zapisy ---
    top_pred.to_csv("predictions.csv", index=False)
    coupons = generate_coupons(top_pred)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)
    
    print(f"Agent finished successfully. Football: {len(football)}, Basketball: {len(basketball)}")
    print(f"Predictions saved: {len(top_pred)} rows, Coupons saved: {len(coupons)}")

if __name__ == "__main__":
    run()
