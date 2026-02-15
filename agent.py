import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from data_loader import load_football_data, upcoming_matches
from predictor import predict_markets

MAX_BETS_PER_MATCH = 3
MAX_BETS_PER_COUPON = 5
MIN_PROB = 0.55

COUPONS_FILE = "coupons.json"
STATUS_FILE = "data_status.json"


def confidence_score(bets):
    probs = [b["Probability"] for b in bets]
    return round(sum(probs) / len(probs) * 100, 1)


def value_flag(prob, odds):
    if odds is None or odds <= 1:
        return False
    return prob > (1 / odds)


def generate_coupons(preds):
    coupons = []
    current = []

    for (_, home, away), g in preds.groupby(["Date", "HomeTeam", "AwayTeam"]):
        g = g.sort_values("Probability", ascending=False).head(MAX_BETS_PER_MATCH)
        for _, r in g.iterrows():
            if r["Probability"] < MIN_PROB:
                continue

            bet = {
                "Date": str(r["Date"].date()),
                "Match": f"{home} vs {away}",
                "League": r["League"],
                "Market": r["Market"],
                "Probability": round(r["Probability"], 3),
                "Odds": r.get("Odds"),
                "Value": value_flag(r["Probability"], r.get("Odds")),
                "ModelAccuracy": round(r["ModelAccuracy"], 3)
            }

            current.append(bet)

            if len(current) == MAX_BETS_PER_COUPON:
                coupons.append({
                    "GeneratedAt": datetime.utcnow().isoformat(),
                    "Confidence": confidence_score(current),
                    "Bets": current
                })
                current = []

    return coupons


def run():
    df, missing_leagues = load_football_data()
    upcoming = upcoming_matches(df)

    preds = predict_markets(upcoming)
    coupons = generate_coupons(preds)

    with open(COUPONS_FILE, "w") as f:
        json.dump(coupons, f, indent=2)

    with open(STATUS_FILE, "w") as f:
        json.dump(missing_leagues, f, indent=2)

    print(f"[INFO] Coupons: {len(coupons)}")
    print(f"[INFO] Missing leagues: {len(missing_leagues)}")


if __name__ == "__main__":
    run()
