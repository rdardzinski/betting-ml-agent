import os
import pickle
import numpy as np
import pandas as pd

MODELS_DIR = "models"

def load_model(market="Over25"):
    """Wczytuje model + metadane z dysku"""
    filename = os.path.join(MODELS_DIR, f"model_{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!"
        )
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"], state["features"]


def predict(df: pd.DataFrame):
    """
    Multi-market prediction dla piłki nożnej
    Zwraca df z kolumnami: *_Prob, *_Confidence, *_ValueFlag, *_ModelAccuracy
    """
    predictions = df.copy()
    markets = ["Over25", "BTTS", "1HGoals", "2HGoals", "Cards", "Corners"]

    for market in markets:
        try:
            model, accuracy, features = load_model(market)

            # Sprawdź czy wszystkie feature'y są w df
            missing = [f for f in features if f not in df.columns]
            if missing:
                print(f"[WARN] Market {market}: brakujące feature'y {missing}")
                features = [f for f in features if f in df.columns]

            X = df[features]

            # predict_proba z obsługą jedynej klasy
            probs_raw = model.predict_proba(X)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(X),), probs_raw[0, 0])
            else:
                probs = probs_raw[:, 1]

            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs * 100).round(1)
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = accuracy

        except FileNotFoundError:
            print(f"[WARN] Model {market} not found, pomijam")
        except Exception as e:
            print(f"[ERROR] Prediction failed for {market}: {e}")

    return predictions, markets
