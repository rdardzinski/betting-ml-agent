import os
import pandas as pd
from datetime import datetime, timedelta

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# Ustawienia
# =========================

MODEL_PATH = "models"
RETENTION_MONTHS = 6
MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

# =========================
# Załaduj dane piłki nożnej
# =========================

print("[INFO] Loading football data...")
football = get_next_matches()

if football.empty:
    raise ValueError("[ERROR] No football matches loaded!")

cutoff_date = datetime.now() - timedelta(days=30*RETENTION_MONTHS)
football["Date"] = pd.to_datetime(football["Date"], errors="coerce")
football = football[football["Date"] >= cutoff_date]
print(f"[INFO] Matches after cutoff ({RETENTION_MONTHS} months): {len(football)}")

# =========================
# Feature engineering
# =========================

football = build_features(football)

# =========================
# Trening modeli
# =========================

for market in MARKETS:
    if market not in football.columns:
        # Tworzymy kolumnę binarną jako proxy
        football[market] = 0
    print(f"[TRAIN] {market}")
    try:
        train_and_save(football, market, model_path=MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Training {market} failed: {e}")

print("[INFO] Training finished successfully!")

# =========================
# Koszykówka (zakomentowane)
# =========================

# from data_loader_basketball import get_basketball_games
# basketball = get_basketball_games()
# if not basketball.empty:
#     basketball = build_features_basketball(basketball)
#     train_and_save_basketball_models(basketball)
