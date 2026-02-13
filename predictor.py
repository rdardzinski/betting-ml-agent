import os
import pickle
import numpy as np

def load_model(market="Over25"):
    filename = f"agent_state_{market}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!")
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    """
    Multi-market prediction
    Zabezpieczenie gdy model.predict_proba zwraca tylko jedną kolumnę
    """
    predictions = df.copy()
    markets = ["Over25","BTTS"]

    for market in markets:
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

    return predictions
