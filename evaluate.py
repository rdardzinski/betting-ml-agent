import pandas as pd
import os
import json

def evaluate():
    if not os.path.exists("predictions.csv"):
        return None

    df = pd.read_csv("predictions.csv")
    total = len(df)
    value_bets = df[df["ValueFlag"]==True]

    metrics = {
        "Total Predictions": total,
        "Value Bets": len(value_bets)
    }

    # update JSON log
    with open("predictions_log.json","w") as f:
        json.dump(metrics,f)

    return metrics
