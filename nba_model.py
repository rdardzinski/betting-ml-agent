import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_nba_model(df):
    df["HomeWin"] = (df["HomeScore"] > df["AwayScore"]).astype(int)

    X = df[["HomeScore","AwayScore"]]
    y = df["HomeWin"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)

    return model

def predict_nba(df, model):
    X = df[["HomeScore","AwayScore"]]
    df["HomeWin_Prob"] = model.predict_proba(X)[:,1]
    return df
