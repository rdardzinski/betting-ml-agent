import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save(features, target, market):
    """
    Trenuje RandomForestClassifier i zapisuje model wraz z dokładnością.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    filename = os.path.join(MODEL_DIR, f"model_{market}.pkl")
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "accuracy": acc}, f)

    print(f"[INFO] Trained model for {market} saved. Accuracy={acc:.3f}")
    return model, acc


def load_model(market="Over25"):
    """
    Wczytuje zapisany model z dysku.
    """
    filename = os.path.join(MODEL_DIR, f"model_{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw run_training.py!")

    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]


def predict(features):
    """
    Przewiduje prawdopodobieństwa wszystkich rynków.
    Zwraca DataFrame z kolumnami Prob, Confidence i ValueFlag dla każdego rynku.
    """
    import pandas as pd
    import numpy as np

    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    predictions = pd.DataFrame(index=features.index)

    for market in markets:
        try:
            model, acc = load_model(market)
            probs_raw = model.predict_proba(features)

            # bezpieczne predict_proba
            if probs_raw.shape[1] == 1:
                probs = np.full((len(features),), probs_raw[0,0])
            else:
                probs = probs_raw[:,1]

            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = acc

        except FileNotFoundError:
            print(f"[WARN] Model {market} not found, skipping predictions.")

    return predictions
