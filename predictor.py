import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Ładowanie modelu
# =========================
def load_model(market="Over25"):
    filename = f"models/agent_state_{market}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!")
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

# =========================
# Trening modelu i zapis
# =========================
def train_and_save(df: pd.DataFrame, market: str, model_path="models"):
    """
    Trenuje model RandomForest dla danego rynku i zapisuje stan.
    """
    os.makedirs(model_path, exist_ok=True)

    # cechy
    feature_cols = [
        "FTHG","FTAG",
        "HomeRollingGoals","AwayRollingGoals",
        "HomeRollingConceded","AwayRollingConceded",
        "HomeForm","AwayForm"
    ]
    X = df[feature_cols]
    y = df[market]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Zapis
    state = {"model": model, "accuracy": acc}
    filename = os.path.join(model_path, f"agent_state_{market}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(state, f)

    print(f"[INFO] Model {market} zapisany w {filename} (accuracy={acc:.2f})")
    return model, acc

# =========================
# Predykcja
# =========================
def predict(df: pd.DataFrame):
    """
    Multi-market prediction dla piłki nożnej.
    """
    predictions = df.copy()
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

    for market in markets:
        try:
            model, accuracy = load_model(market)
        except FileNotFoundError:
            print(f"[WARN] Model {market} nie znaleziony, pomijam")
            continue

        features = df[[
            "FTHG","FTAG",
            "HomeRollingGoals","AwayRollingGoals",
            "HomeRollingConceded","AwayRollingConceded",
            "HomeForm","AwayForm"
        ]]

        # predict_proba zabezpieczenie
        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy

    return predictions
