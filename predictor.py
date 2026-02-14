import pickle
import os
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
    oraz brakujące feature'y
    """
    predictions = df.copy()
    markets = ["Over25","BTTS"]

    for market in markets:
        model, accuracy = load_model(market)

        features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]

        # Uzupełnienie brakujących kolumn
        for f in features:
            if f not in predictions.columns:
                predictions[f] = 0

        X = predictions[features]

        # bezpieczne predict_proba
        probs_raw = model.predict_proba(X)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(X),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy

    return predictions
