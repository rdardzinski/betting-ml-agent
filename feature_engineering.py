import pandas as pd
from elo import compute_elo

FEATURES = [
    "HomeGoalsAvg","AwayGoalsAvg",
    "HomeConcededAvg","AwayConcededAvg",
    "HomeElo","AwayElo",
    "OddsHome","OddsDraw","OddsAway"
]

def build_features(df):
    df = df.sort_values("Date")

    for side in ["Home","Away"]:
        df[f"{side}GoalsAvg"] = (
            df.groupby(f"{side}Team")["FTHG" if side=="Home" else "FTAG"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df[f"{side}ConcededAvg"] = (
            df.groupby(f"{side}Team")["FTAG" if side=="Home" else "FTHG"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )

    df = compute_elo(df)
    return df
