# agent.py
import json
import pandas as pd
from pathlib import Path

PREDICTIONS_FILE = "predictions.csv"
COUPONS_FILE = "coupons.json"

MAX_BETS_PER_COUPON = 5
MIN_PROB = 0.55

MARKETS = [
    "Over25",
    "BTTS",
    "1HGoals",
    "2HGoals",
    "Cards",
    "Corners",
]


def load_predictions() -> pd.DataFrame:
    if not Path(PREDICTIONS_FILE).exists():
        raise FileNotFoundError("predictions.csv not found – uruchom run_training.py")

    df = pd.read_csv(PREDICTIONS_FILE)

    base_cols = ["Date", "HomeTeam", "AwayTeam", "League"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = "Unknown"

    rows = []

    # 🔁 TRANSFORMACJA STAREGO FORMATU → DŁUGI FORMAT
    for market in MARKETS:
        prob_col = f"{market}_Prob"
        acc_col = f"{market}_ModelAccuracy"

        if prob_col not in df.columns or acc_col not in df.columns:
            continue

        tmp = df[base_cols + [prob_col, acc_col]].copy()
        tmp.rename(
            columns={
                prob_col: "Probability",
                acc_col: "ModelAccuracy",
            },
            inplace=True,
        )
        tmp["Market"] = market
        rows.append(tmp)

    if not rows:
        raise ValueError("Brak rozpoznawalnych rynków w predictions.csv")

    long_df = pd.concat(rows, ignore_index=True)

    return long_df


def build_coupons(df: pd.DataFrame) -> list:
    df = df[df["Probability"] >= MIN_PROB].copy()

    df["Score"] = df["Probability"] * df["ModelAccuracy"]
    df = df.sort_values("Score", ascending=False)

    coupons = []
    used_matches = set()
    current_coupon = []

    for _, row in df.iterrows():
        match_id = f"{row['Date']}|{row['HomeTeam']}|{row['AwayTeam']}"

        if match_id in used_matches:
            continue

        bet = {
            "Date": row["Date"],
            "Match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
            "League": row["League"],
            "Market": row["Market"],
            "Probability": round(row["Probability"], 3),
            "ModelAccuracy": round(row["ModelAccuracy"], 3),
            "ValueFlag": row["Probability"] >= 0.55,
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

    print(f"[INFO] Predictions loaded: {len(df)} rows")

    print("[INFO] Building coupons...")
    coupons = build_coupons(df)

    print(f"[INFO] Coupons created: {len(coupons)}")
    save_coupons(coupons)

    print("[DONE] Agent finished successfully")


if __name__ == "__main__":
    run()
