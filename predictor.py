import pickle

def load_model():
    with open("agent_state.pkl","rb") as f:
        state = pickle.load(f)
    return state["model"], state["accuracy"]

def predict(df):
    model, accuracy = load_model()

    features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]
    probs = model.predict_proba(features)[:,1]

    df["Over25_Prob"] = probs
    df["Confidence"] = (probs*100).round(1)
    df["ValueFlag"] = probs > 0.55
    df["ModelAccuracy"] = accuracy
    return df
