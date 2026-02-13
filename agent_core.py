import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"

# ===============================
# DATA
# ===============================

def load_data():
    df = pd.read_csv(DATA_URL)
    df = df[['HomeTeam','AwayTeam','FTHG','FTAG','FTR']]
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
    df['Under25'] = (df['TotalGoals'] <= 2.5).astype(int)
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    df = df.dropna()
    return df

# ===============================
# TRAIN MULTI-MARKET MODELS
# ===============================

def train_models(df):
    X = df[['FTHG','FTAG']]

    markets = {
        "Over25": df['Over25'],
        "Under25": df['Under25'],
        "BTTS": df['BTTS']
    }

    results = {}

    for name, target in markets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, random_state=42
        )

        model = XGBClassifier()
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, f"model_{name}.pkl")
        results[name] = acc

    return results

# ===============================
# PREDICTIONS + VALUE
# ===============================

def predict_markets(df):
    selections = []

    for market in ["Over25","Under25","BTTS"]:
        model = joblib.load(f"model_{market}.pkl")
        prob = model.predict_proba(df[['FTHG','FTAG']])[:,1]

        df_temp = df.copy()
        df_temp["Market"] = market
        df_temp["Probability"] = prob

        # przykładowy kurs symulowany
        df_temp["Odds"] = 1.80

        df_temp["EV"] = (df_temp["Probability"] * df_temp["Odds"]) - 1

        selections.append(df_temp)

    all_sel = pd.concat(selections)
    all_sel = all_sel[all_sel["EV"] > 0.05]  # dynamiczny próg value

    return all_sel.sort_values("EV", ascending=False)
