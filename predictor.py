import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Konfiguracja rynków i cech
# =========================
FEATURE_COLS = [
    "FTHG", "FTAG",
    "HomeRollingGoals", "AwayRollingGoals",
    "HomeRollingConceded", "AwayRollingConceded",
    "HomeForm", "AwayForm"
]

MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

# =========================
# Funkcja trenowania i zapisu modelu
# =========================
def train_and_save(df: pd.DataFrame, market: str, model_path="models"):
    """
    Trenuje RandomForest dla danego rynku i zapisuje model wraz z dokładnością i listą cech.
    """
    os.makedirs(model_path, exist_ok=True)

    # X = cechy, y = target dla danego rynku
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Brak kolumn {missing_cols} w danych, wypełniam zerami")
        for c in missing_cols:
            df[c] = 0

    X = df[FEATURE_COLS]
    y = df[market]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # zapis modelu
    state = {"model": model, "accuracy": acc, "feature_cols": FEATURE_COLS}
    filename = os.path.join(model_path, f"agent_state_{market}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(state, f)

    print(f"[INFO] Model {market} zapisany w {filename} (accuracy={acc:.2f})")
    return model, acc

# =========================
# Funkcja ładowania modelu
# =========================
def load_model(market="Over25", model_path="models"):
    filename = os.path.join(model_path, f"agent_state_{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!")
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"], state["feature_cols"]

# =========================
# Funkcja predykcji
# =========================
def predict(df: pd.DataFrame):
    """
    Multi-market prediction dla piłki nożnej.
    Zabezpiecza brakujące kolumny i dopasowuje cechy zgodnie z modelem.
    """
    preds = df.copy()

    for market in MARKETS:
        try:
            model, accuracy, feature_cols = load_model(market)
        except FileNotFoundError:
            print(f"[WARN] Model {market} nie znaleziony, pomijam")
            continue

        # uzupełnienie brakujących kolumn
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"[WARN] Brak kolumn {missing_cols} dla {market}, wypełniam 0")
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
