import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import get_next_matches
from feature_engineering import build_features

# =========================
# KONFIGURACJA
# =========================

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CUTOFF_DATE = datetime.now() - timedelta(days=180)  # ostatnie 6 miesięcy

MARKETS = {
    "Over25": {
        "target": "Over25",
    },
    "BTTS": {
        "target": "BTTS",
    },
    "1HGoals": {
        "target": "Over05_1H",
    },
    "2HGoals": {
        "target": "Over05_2H",
    },
    "Cards": {
        "target": "Over35_Cards",
    },
    "Corners": {
        "target": "Over85_Corners",
    },
}

BASE_FEATURES = [
    "FTHG",
    "FTAG",
    "HomeRollingGoals",
    "AwayRollingGoals",
    "HomeRollingConceded",
    "AwayRollingConceded",
    "HomeForm",
    "AwayForm",
]

# =========================
# LOAD + FEATURES
# =========================

print("[INFO] Loading football data...")
df = get_next_matches()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df[df["Date"] >= CUTOFF_DATE]

print(f"[INFO] Matches after cutoff: {len(df)}")

df = build_features(df)

# =========================
# TRENING
# =========================

for market, cfg in MARKETS.items():
    print(f"[TRAIN] {market}")

    target = cfg["target"]

    if target not in df.columns:
        print(f"[WARN] Target {target} not found – skipping")
        continue

    # 👉 tylko istniejące feature’y
    features = [f for f in BASE_FEATURES if f in df.columns]

    if len(features) < 2:
        print(f"[WARN] Not enough features for {market}")
        continue

    data = df[features + [target]].dropna()

    if len(data) < 100:
        print(f"[WARN] Not enough rows for {market}")
        continue

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(
        {
            "model": model,
            "accuracy": acc,
            "features": features,   # 🔥 KLUCZOWE
            "trained_at": datetime.now(),
        },
        f"{MODELS_DIR}/model_{market}.pkl",
    )

    print(f"[INFO] Model saved: {market} (acc={acc:.2f})")

print("[DONE] Training finished successfully")
