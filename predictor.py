import os
import pickle

def load_model(market="Over25"):
    filename = f"agent_state_{market}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} nie istnieje. Uruchom najpierw notebook retrainingowy!")
    with open(filename,"rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    # multi-market
    predictions = df.copy()
    markets = ["Over25","BTTS"]
    for market in markets:
        model, accuracy = load_model(market)
        features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]
        probs = model.predict_proba(features)[:,1]
        predictions[f"{market}_Prob"] = probs
        predictions[f"{market}_Confidence"] = (probs*100).round(1)
        predictions[f"{market}_ValueFlag"] = probs > 0.55
        predictions[f"{market}_ModelAccuracy"] = accuracy
    return predictions
