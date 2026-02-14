import os
import sys
import pandas as pd
from datetime import datetime

# Dodanie katalogu głównego repo do PYTHONPATH (notebooks/run_training.py -> ../)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save
import numpy as np

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
# Przygotowanie targetów (dummy dla treningu)
# =========================
np.random.seed(42)
for market in MARKETS:
    if market not in football_features.columns:
        football_features[market] = np.random.randint(0,2,len(football_features))

# =========================
# Trening i zapis modeli
# =========================
train_and_save(football_features)
print("[INFO] Training finished successfully!")
