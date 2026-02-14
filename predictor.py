import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# FUNKCJA: trenowanie i zapis modelu
# =========================
def train_and_save(df: pd.DataFrame, features: list, target: str, market: str):
    """
    Trenuje RandomForestClassifier na danych df[features] -> df[target]
    Zapisuje model w models/agent_state_{market}.pkl
    """
    os.makedirs("models", exist_ok=True)

    # usuń wiersze z brakującymi danymi
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # zapis modelu
    filename = f"models/agent_state_{market}.pkl"
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "accuracy": acc}, f)

    print(f"[INFO] Model {market} saved: accuracy={acc:.3f}")
    return model, acc


# =========================
# FUNKCJA: predykcja na nowych danych
# =========================
def predict(df: pd.DataFrame):
    """
    Przygotowanie predykcji dla wszystkich rynków piłki nożnej
    Zwraca df z kolumnami:
        {market}_Prob, {market}_Confidence, {market}_ValueFlag, {market}_ModelAccuracy
    """
    df = df.copy()
    markets = ["Over25", "BTTS", "1HGoals", "2HGoals", "Cards", "Corners"]

    for market in markets:
        filename = f"models/agent_state_{market}.pkl"
        if not os.path.exists(filename):
            print(f"[WARN] Model {market} not found, pomijam.")
            continue

        with open(filename, "rb") as f:
            state = pickle.load(f)

        model = state["model"]
        accuracy = state.get("accuracy", 0.5)

        # przygotowanie cech do predykcji
        feature_cols = [col for col in df.columns if col in model.feature_names_in_]
        if not feature_cols:
            print(f"[WARN] Brak zgodnych cech dla {market}, pomijam.")
            continue

        X = df[feature_cols]

        # bezpieczne predict_proba
        try:
            probs_raw = model.predict_proba(X)
            if probs_raw.shape[1] == 1:
                probs = np.full((len(X),), probs_raw[0, 0])
            else:
                probs = probs_raw[:, 1]
        except Exception as e:
            print(f"[ERROR] Predykcja dla {market} nie powiodła się: {e}")
            probs = np.full(len(X), 0.5)

        df[f"{market}_Prob"] = probs
        df[f"{market}_Confidence"] = (probs * 100).round(1)
        df[f"{market}_ValueFlag"] = probs > 0.55
        df[f"{market}_ModelAccuracy"] = accuracy

    return df
