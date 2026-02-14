import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# Parametry
# =========================
MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
CUTOFF_DATE = datetime.today() - timedelta(days=180)  # ostatnie 6 miesięcy

# =========================
# Pobranie danych piłki nożnej
# =========================
print("[INFO] Loading football data...")
football = get_next_matches()
if football.empty:
    print("[ERROR] Brak danych piłki nożnej.")
    exit(1)

# konwersja daty i filtrowanie ostatnich 6 miesięcy
football["Date"] = pd.to_datetime(football["Date"], errors="coerce")
football = football[football["Date"] >= CUTOFF_DATE]
print(f"[INFO] Matches after cutoff: {len(football)}")

# =========================
# Budowa cech
# =========================
football = build_features(football)

# =========================
# Trenowanie modeli
# =========================
for market in MARKETS:
    target_col = market  # oczekuje, że kolumna target istnieje w football
    if target_col not in football.columns:
        print(f"[WARN] Brak kolumny target dla rynku {market}, pomijam.")
        continue

    try:
        train_and_save(football, target_col, market)
    except Exception as e:
        print(f"[ERROR] Trening dla rynku {market} nie powiódł się: {e}")
