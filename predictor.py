import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "models"

MARKETS = ["Over25", "BTTS", "1HGoals", "2HGoals", "Cards", "Corners"]

def load_model(market="Over25"):
    filename = os.path.join(MODEL_DIR, f"{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model dla {market} nie znaleziony: {filename}")
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df: pd.DataFrame):
    """
    Przewidywanie dla wielu rynków jednocześnie.
    Zachowuje kolumny wejściowe, dodaje prob, ValueFlag, ModelAccuracy.
    """
    predictions = df.copy()

    for market in MARKETS:
        try:
            model, accuracy = load_model(market)
            features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]].copy()

            # bezpieczne predict_proba
            probs_raw = model.predict_proba(features)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(features),), probs_raw[0,0])
            else:
                probs = probs_raw[:,1]

            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = accuracy
        except FileNotFoundError:
            # brak modelu – wstaw wartości domyślne
            predictions[f"{market}_Prob"] = 0.5
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = 0.0
        except Exception as e:
            print(f"[ERROR] Prediction failed for {market}: {e}")
            predictions[f"{market}_Prob"] = 0.5
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = 0.0

    return predictions, MARKETS
