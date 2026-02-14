import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_loader import get_next_matches
from feature_engineering import build_features

# =========================
# KONFIGURACJA
# =========================

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CUTOFF_DATE = datetime.now() - timedelta(days=180)

TARGETS = {
    # klasyczne
    "Over25": lambda df: (df["FTHG"] + df["FTAG"] > 2).astype(int),
    "BTTS": lambda df: ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int),

    # połowy
    "1H_Over05": lambda df: (df["HTHG"] + df["HTAG"] > 0).astype(int),
    "2H_Over15": lambda df: ((df["FTHG"] + df["FTAG"]) - (df["HTHG"] + df["HTAG"]) > 1).astype(int),

    # drużynowe
    "HomeScore": lambda df: (df["FTHG"] > 0).astype(int),
    "AwayScore": lambda df: (df["FTAG"] > 0).astype(int),

    # sumy
    "Over15": lambda df: (df["FTHG"] + df["FTAG"] > 1).astype(int),
    "Over35": lambda df: (df["FTHG"] + df["FTAG"] > 3).astype(int),
}

FEATURE_COLUMNS = [
    "HomeRollingGoals",
    "AwayRollingGoals",
    "HomeRollingConceded",
    "AwayRollingConceded",
    "HomeForm",
    "AwayForm"
]

# =========================
# TRAINING
# =========================

def train_model(df, target_name, target_func):
    print(f"[TRAIN] {target_name}")

    df = df.copy()
    df[target_name] = target_func(df)

    df = df.dropna(subset=FEATURE_COLUMNS + [target_name])

    X = df[FEATURE_COLUMNS]
    y = df[target_name]

    if len(X) < 200:
        print(f"[SKIP] Not enough samples for {target_name}")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_path = f"{MODELS_DIR}/{target_name}.joblib"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model.fit(X_train, y_train)
        print(f"[INFO] Incremental retrain for {target_name}")
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"[INFO] New model trained for {target_name}")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, model_path)
    print(f"[OK] {target_name} accuracy: {round(acc, 3)}")

    return acc

# =========================
# MAIN
# =========================

def main():
    print("[INFO] Loading football data...")
    df = get_next_matches()

    if df.empty:
        raise RuntimeError("No football data loaded")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"] >= CUTOFF_DATE]

    print(f"[INFO] Matches after cutoff: {len(df)}")

    df = build_features(df)

    results = {}

    for target, func in TARGETS.items():
        acc = train_model(df, target, func)
        if acc:
            results[target] = acc

    print("\n====== TRAINING SUMMARY ======")
    for k, v in results.items():
        print(f"{k}: {round(v,3)}")

    print("[SUCCESS] Training finished")

    # -----------------------------
    # 🏀 BASKETBALL – WYŁĄCZONE
    # -----------------------------
    """
    from data_loader_basketball import get_basketball_games
    basketball = get_basketball_games()
    """

if __name__ == "__main__":
    main()
