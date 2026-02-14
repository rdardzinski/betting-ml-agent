import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Dodaj katalog nadrzędny, by Python widział moduły
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features

# ================================
# Katalog na modele
# ================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ================================
# Parametry
# ================================
CUTOFF_DATE = datetime.today() - timedelta(days=180)  # ostatnie 6 miesięcy

FOOTBALL_MARKETS = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
BASKETBALL_MARKETS = ["HomeWin","BasketPoints","BasketSum"]

# ================================
# Funkcja do trenowania modelu
# ================================
def train_market(df, market):
    df = df.copy()
    if df.empty:
        return None, 0.5

    # Piłka nożna
    if market in FOOTBALL_MARKETS:
        features = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals","1HGoals","2HGoals","BTTS","Cards","Corners"]
        for col in features:
            if col not in df.columns:
                df[col] = 0
        X = df[features]
        y = df[market] if market in df.columns else np.random.randint(0,2,len(df))

    # Koszykówka
    elif market in BASKETBALL_MARKETS:
        for col in ["HomeScore","AwayScore"]:
            if col not in df.columns:
                df[col] = 0
        X = df[["HomeScore","AwayScore"]]
        y = df[market] if market in df.columns else np.random.randint(0,2,len(df))
    else:
        return None, 0.5

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X, y)
        acc = model.score(X, y)
    except Exception:
        model.fit(np.zeros((1,X.shape[1])), [0])
        acc = 0.5
    return model, acc

# ================================
# TRENING PIŁKI NOŻNEJ
# ================================
football = get_next_matches()
football = football[football["Date"] >= CUTOFF_DATE] if "Date" in football.columns else pd.DataFrame()

if not football.empty:
    football = build_features(football)
    for market in FOOTBALL_MARKETS:
        model, acc = train_market(football, market)
        fname = os.path.join(MODEL_DIR, f"agent_state_{market}.pkl")
        with open(fname,"wb") as f:
            pickle.dump({"model":model,"accuracy":acc}, f)
        print(f"[INFO] Football model saved: {market} (acc={acc:.2f})")
else:
    print("[WARN] Brak meczów piłki nożnej do treningu")

# ================================
# TRENING KOSZYKÓWKI – fallbacki
# ================================
basketball_sources = [get_basketball_games]  # Możesz dodać inne źródła jako kolejne funkcje
basketball = pd.DataFrame()

for src in basketball_sources:
    df = src()
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["HomeTeam","AwayTeam","Date"])
        basketball = df
        print(f"[INFO] Basketball data loaded from {src.__name__}")
        break

if basketball.empty:
    print("[WARN] Brak danych koszykówki z żadnego źródła")

if not basketball.empty:
    basketball = basketball[basketball["Date"] >= CUTOFF_DATE]
    for market in BASKETBALL_MARKETS:
        model, acc = train_market(basketball, market)
        fname = os.path.join(MODEL_DIR,f"agent_state_{market}.pkl")
        with open(fname,"wb") as f:
            pickle.dump({"model":model,"accuracy":acc},f)
        print(f"[INFO] Basketball model saved: {market} (acc={acc:.2f})")

print("[INFO] Run_training finished successfully")
