import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_COLS = [
    "FTHG", "FTAG",
    "HomeRollingGoals", "AwayRollingGoals",
    "HomeRollingConceded", "AwayRollingConceded",
    "HomeForm", "AwayForm"
]

MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

def train_and_save(df: pd.DataFrame, market: str, model_path="models"):
    os.makedirs(model_path, exist_ok=True)

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    for c in missing_cols:
        df[c] = 0

    X = df[FEATURE_COLS]
    y = df[market]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    state = {"model": model, "accuracy": acc, "feature_cols": FEATURE_COLS}
    filename = os.path.join(model_path, f"agent_state_{market}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(state, f)

    print(f"[INFO] Model {market} saved ({acc:.2f})")
    return model, acc

def load_model(market="Over25", model_path="models"):
    filename = os.path.join(model_path, f"agent_state_{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run training notebook first!")
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"], state["feature_cols"]

def predict(df: pd.DataFrame):
    preds = df.copy()
    for market in MARKETS:
        try:
            model, accuracy, feature_cols = load_model(market)
        except FileNotFoundError:
            print(f"[WARN] Model {market} not found, skipping")
            continue

        missing_cols = [c for c in feature_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = 0

        X = df[feature_cols]
        probs_raw = model.predict_proba(X)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(X),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        preds[f"{market}_Prob"] = probs
        preds[f"{market}_Confidence"] = (probs*100).round(1)
        preds[f"{market}_ValueFlag"] = probs > 0.55
        preds[f"{market}_ModelAccuracy"] = accuracy
    return preds
