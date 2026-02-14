import pandas as pd
import os

HISTORY_FILE = "history/predictions_history.csv"

def update_history(predictions: pd.DataFrame):
    os.makedirs("history", exist_ok=True)

    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([hist, predictions], ignore_index=True)
    else:
        combined = predictions.copy()

    combined.to_csv(HISTORY_FILE, index=False)
    return combined


def evaluate_over25(history: pd.DataFrame):
    df = history.dropna(subset=["FTHG", "FTAG", "Over25"])
    df["Result"] = (df["FTHG"] + df["FTAG"] > 2).astype(int)
    df["Hit"] = (df["Over25"] > 0.55) & (df["Result"] == 1)

    stake = 1
    odds = 1.85  # uśrednione
    df["Profit"] = df["Hit"] * (odds - 1) - stake * (~df["Hit"])

    return {
        "bets": len(df),
        "hit_rate": df["Hit"].mean(),
        "roi": df["Profit"].sum() / len(df)
    }
