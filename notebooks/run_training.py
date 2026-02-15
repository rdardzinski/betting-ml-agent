import sys
from pathlib import Path

# =========================
# FIX IMPORT PATH
# =========================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from datetime import datetime, timedelta

from data_loader import load_football_data
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# PARAMETRY
# =========================
MONTHS_BACK = 6
CUTOFF_DATE = datetime.utcnow() - timedelta(days=30 * MONTHS_BACK)

MARKETS = [
    "Over25",
    "BTTS",
    "1HGoals",
    "2HGoals",
    "Cards",
    "Corners",
]

# =========================
# PIPELINE
# =========================
def main():
    print("[INFO] Loading football matches...")
    df = load_football_data(months_back=MONTHS_BACK)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= CUTOFF_DATE]

    print(f"[INFO] Matches after cutoff ({MONTHS_BACK} months): {len(df)}")

    if df.empty:
        raise RuntimeError("No football data after cutoff date")

    print("[INFO] Building features...")
    features = build_features(df)

    if features.empty:
        raise RuntimeError("Feature engineering returned empty DataFrame")

    # =========================
    # TRAIN MODELS PER MARKET
    # =========================
    for market in MARKETS:
        print(f"[TRAIN] {market}")

        if market not in features.columns:
            print(f"[WARN] Market {market} missing in data, creating dummy target.")
            features[market] = 0

        X = features.drop(columns=MARKETS, errors="ignore")
        y = features[market]

        train_and_save(
            X=X,
            y=y,
            market=market,
        )

    print("[SUCCESS] Training finished successfully")


if __name__ == "__main__":
    main()
