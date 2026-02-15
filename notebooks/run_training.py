import sys
import os

# 🔧 DODAJ ROOT PROJEKTU DO PYTHON PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from datetime import datetime, timedelta
import pandas as pd

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# CONFIG
# =========================
CUTOFF_DATE = datetime.now() - timedelta(days=180)
MARKETS = ["Over25", "BTTS"]

# =========================
# LOAD DATA
# =========================
print("[INFO] Loading football data...")
df = get_next_matches()

df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= CUTOFF_DATE]

print(f"[INFO] Matches after cutoff (6 months): {len(df)}")

# =========================
# FEATURE ENGINEERING
# =========================
df = build_features(df)

# =========================
# TRAIN PER LEAGUE & MARKET
# =========================
for league, league_df in df.groupby("League"):
    print(f"[INFO] Training league: {league}")

    for market in MARKETS:
        print(f"[TRAIN] {league} | {market}")

        try:
            train_and_save(
                df=league_df,
                market=market,
                league=league
            )
        except Exception as e:
            print(f"[ERROR] Training failed for {league} {market}: {e}")

print("[INFO] Training finished successfully")
