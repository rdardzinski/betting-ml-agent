import pandas as pd

def detect_regime(df):
    regime = (
        df.groupby("League")["TotalGoals"]
        .mean()
        .reset_index()
    )

    regime["Regime"] = regime["TotalGoals"].apply(
        lambda x: "OVER" if x > 2.7 else "UNDER"
    )

    return regime
