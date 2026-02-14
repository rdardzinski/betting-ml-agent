import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

def train_and_save(df):
    """
    Trenuje modele dla każdego rynku i zapisuje jako pickle
    """
    features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals",
                "HomeRollingConceded","AwayRollingConceded","HomeForm","AwayForm"]

    for market in MARKETS:
        target = market
        if target not in df.columns:
            continue

        X = df[features]
        y = df[target]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        acc = model.score(X, y)

        filename = os.path.join(MODEL_DIR, f"agent_state_{market}.pkl")
        with open(filename, "wb") as f:
            pickle.dump({"model":model, "accuracy":acc}, f)
        print(f"[INFO] Football model saved: {market} (acc={acc:.2f})")

def load_model(market="Over25"):
    filename = os.path.join(MODEL_DIR, f"agent_state_{market}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw trening!")
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    """
    Przewiduje prawdopodobieństwa dla wszystkich rynków bukmacherskich.
    Zwraca df z kolumnami: Market_Prob, Market_ValueFlag, Market_ModelAccuracy
    """
    df = df.copy()
    features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals",
                "HomeRollingConceded","AwayRollingConceded","HomeForm","AwayForm"]

    for market in MARKETS:
        try:
            model, accuracy = load_model(market)
        except FileNotFoundError:
            print(f"[WARN] Model {market} not found, pomijam")
            continue

        X = df[features]
        probs_raw = model.predict_proba(X)
        if probs_raw.shape[1] == 1:
            probs = np.full((len(X),), probs_raw[0,0])
        else:
            probs = probs_raw[:,1]

        df[f"{market}_Prob"] = probs
        df[f"{market}_Confidence"] = (probs*100).round(1)
        df[f"{market}_ValueFlag"] = probs > 0.55
        df[f"{market}_ModelAccuracy"] = accuracy
    return df
