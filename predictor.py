import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =========================
# Funkcja wczytująca model dla danego rynku
# =========================
def load_model(market="Over25"):
    filename = f"models/model_{market}.pkl"
    if not os.path.exists(filename):
        print(f"[WARN] Model {market} not found, pomijam")
        return None, None
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state.get("accuracy", 0.5)

# =========================
# Funkcja predykcji dla wielu rynków
# =========================
def predict(df, markets=["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]):
    """
    df: pd.DataFrame z kolumnami wymaganymi do feature engineering
    markets: lista rynków do predykcji
    Zwraca df z Prob, ValueFlag, ModelAccuracy dla każdego rynku
    """
    predictions = df.copy()

    for market in markets:
        model, accuracy = load_model(market)
        if model is None:
            # brak modelu – ustaw domyślne wartości
            predictions[f"{market}_Prob"] = 0.5
            predictions[f"{market}_ValueFlag"] = False
            predictions[f"{market}_ModelAccuracy"] = 0.5
            continue

        # Wybór cech – wszystkie liczby w df, bez nazw drużyn i lig
        feature_cols = [c for c in df.columns if c not in ["HomeTeam","AwayTeam","League","Date"]]
        features = df[feature_cols].fillna(0)

        # predict_proba, zabezpieczenie przed jedną klasą
        probs_raw = model.predict_proba(features)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(features),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy

    return predictions, markets

# =========================
# Funkcja do trenowania modelu i zapisu
# =========================
def train_and_save(df, target_col, market_name):
    """
    df: DataFrame z feature_cols i target_col
    target_col: kolumna binarna (0/1) dla rynku
    market_name: nazwa rynku, np. "Over25"
    """
    from sklearn.model_selection import train_test_split
    feature_cols = [c for c in df.columns if c not in ["HomeTeam","AwayTeam","League","Date"]]
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    if y.nunique() < 2:
        print(f"[WARN] Za mało klas w {market_name}, pomijam trening")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # zapis modelu
    os.makedirs("models", exist_ok=True)
    with open(f"models/model_{market_name}.pkl", "wb") as f:
        pickle.dump({"model": model, "accuracy": acc}, f)

    print(f"[INFO] Model {market_name} zapisany (acc={acc:.2f})")
    return model, acc
