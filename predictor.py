import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================
# Ładowanie modelu
# =========================

def load_model(market="Over25"):
    filename = f"models/agent_state_{market}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw run_training.py")
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

# =========================
# Predykcja
# =========================

def predict(df):
    """
    Multi-market prediction dla piłki nożnej
    """
    predictions = df.copy()
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

    for market in markets:
        model, accuracy = load_model(market)

        features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals","HomeForm","AwayForm"]
        # Uzupełnij brakujące kolumny cech
        for f in features:
            if f not in predictions.columns:
                predictions[f] = pd.Series(0, index=predictions.index)

        X = predictions[features]

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
