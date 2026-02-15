# agent.py
import json
import pandas as pd
from pathlib import Path

PREDICTIONS_FILE = "predictions.csv"
COUPONS_FILE = "coupons.json"

MAX_BETS_PER_COUPON = 5
MIN_PROB = 0.55


def load_predictions() -> pd.DataFrame:
    if not Path(PREDICTIONS_FILE).exists():
        raise FileNotFoundError("predictions.csv not found – uruchom run_training.py")

    df = pd.read_csv(PREDICTIONS_FILE)

    required_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "League",
        "Market",
        "Probability",
        "ModelAccuracy",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w predictions.csv: {missing}")

    return df


def build_coupons(df: pd.DataFrame) -> list:
    # tylko wartościowe zakłady
    df = df[df["Probability"] >= MIN_PROB].copy()

    # sortowanie: najwyższe prawdopodobieństwo + accuracy
    df["Score"] = df["Probability"] * df["ModelAccuracy"]
    df = df.sort_values("Score", ascending=False)

    coupons = []
    used_matches = set()

    current_coupon = []

    for _, row in df.iterrows():
        match_id = f"{row['Date']}|{row['HomeTeam']}|{row['AwayTeam']}"

        # jeden rynek na mecz
        if match_id in used_matches:
            continue

        bet = {
            "Date": row["Date"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "League": row["League"],
            "Market": row["Market"],
            "Probability": round(row["Probability"], 3),
            "ModelAccuracy": round(row["ModelAccuracy"], 3),
        }

        current_coupon.append(bet)
        used_matches.add(match_id)

        if len(current_coupon) == MAX_BETS_PER_COUPON:
            coupons.append(current_coupon)
            current_coupon = []
            used_matches = set()

        if len(coupons) == 5:
            break

    if current_coupon:
        coupons.append(current_coupon)

    return coupons


def save_coupons(coupons: list):
    with open(COUPONS_FILE, "w", encoding="utf-8") as f:
        json.dump(coupons, f, indent=2, ensure_ascii=False)


def run():
    print("[INFO] Loading predictions...")
    df = load_predictions()

    print("[INFO] Building coupons...")
    coupons = build_coupons(df)

    print(f"[INFO] Coupons created: {len(coupons)}")
    save_coupons(coupons)

    print("[DONE] Agent finished successfully")


if __name__ == "__main__":
    run()
