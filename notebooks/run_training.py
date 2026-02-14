import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np

# Dodanie katalogu głównego repo do PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# Parametry
# =========================
RETENTION_MONTHS = 6
TODAY = datetime.today()
CUTOFF_DATE = TODAY - pd.DateOffset(months=RETENTION_MONTHS)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

# =========================
# Ładowanie danych
# =========================
print("[INFO] Loading football data...")
football = get_next_matches()

if football.empty:
    print("[WARN] Brak danych piłki nożnej do treningu!")
    exit(0)

# Konwersja dat
football["Date"] = pd.to_datetime(football["Date"], errors="coerce")
football = football.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

# Retention: ostatnie 6 miesięcy
football_recent = football[football["Date"] >= CUTOFF_DATE].copy()
print(f"[INFO] Matches after cutoff ({RETENTION_MONTHS} months): {len(football_recent)}")

# =========================
# Feature Engineering
# =========================
football_features = build_features(football_recent)

# =========================
# Przygotowanie dummy targetów dla brakujących kolumn rynkowych
# =========================
np.random.seed(42)
for market in MARKETS:
    if market not in football_features.columns:
        print(f"[WARN] Market {market} missing in data, creating dummy target.")
        football_features[market] = np.random.randint(0,2,len(football_features))

# =========================
# Trening i zapis modeli dla każdego rynku
# =========================
feature_cols = [
    "FTHG","FTAG",
    "HomeRollingGoals","AwayRollingGoals",
    "HomeRollingConceded","AwayRollingConceded",
    "HomeForm","AwayForm"
]

for market in MARKETS:
    features = football_features[feature_cols].copy()
    target = football_features[market]

    # Wywołanie train_and_save z wszystkimi wymaganymi argumentami
    train_and_save(features, target, market)
    print(f"[INFO] Model saved: {market}")

print("[INFO] Training finished successfully!")
