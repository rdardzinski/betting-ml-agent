import pickle
import numpy as np

def load_agent():
    with open("agent_state.pkl", "rb") as f:
        return pickle.load(f)

def predict_matches(matches_df):
    agent = load_agent()
    model = agent["model"]

    # placeholder features (docelowo: xG, form, rolling stats)
    matches_df["FTHG"] = 1
    matches_df["FTAG"] = 1

    probs = model.predict_proba(
        matches_df[["FTHG","FTAG"]]
    )[:,1]

    matches_df["Over25_Prob"] = probs
    matches_df["ValueFlag"] = probs > 0.55

    return matches_df
