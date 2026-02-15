import os, joblib, time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_and_save(df, market, league):
    ts = time.strftime("%Y%m%d_%H%M")
    path = f"models/football/{league}/{market}"
    os.makedirs(path, exist_ok=True)

    target_map = {
        "Over25": (df["FTHG"] + df["FTAG"] > 2).astype(int),
        "BTTS": ((df["FTHG"]>0)&(df["FTAG"]>0)).astype(int)
    }

    if market not in target_map:
        return

    X = df.select_dtypes("number")
    y = target_map[market]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump(model, f"{path}/{ts}.joblib")
