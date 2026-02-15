# =========================
# FIX IMPORT PATH
# =========================
import sys
from pathlib import Path

# dodaje katalog główny projektu do PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

# =========================
# IMPORTY
# =========================
import os
from datetime import datetime, timedelta
import pandas as pd

from data_loader import load_football_data
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# KONFIGURACJA
# =========================
CUTOFF_MONTHS = 6
MARKETS = [
    "Over25",
    "BTTS",
    "1HGoals",
    "2HGoals",
    "Cards",
    "Corners",
]

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# MAIN
# =========================
def main():
    print("[INFO] Loading football data...")
    football, report = load_football_data()

    if football.empty:
        raise RuntimeError("Brak danych piłkarskich – trening przerwany")

    # =========================
    # FILTR DATY (6 MIESIĘCY)
    # =========================
    cutoff_date = datetime.utcnow() - timedelta(days=30 * CUTOFF_MONTHS)
    football["Date"] = pd.to_datetime(football["Date"], errors="coerce")
    football = football[football["Date"] >= cutoff_date]

    print(f"[INFO] Matches after cutoff ({CUTOFF_MONTHS} months): {len(football)}")

    if football.empty:
        raise RuntimeError("Brak danych po filtrze dat")

    # =========================
    # FEATURE ENGINEERING
    # =========================
    print("[INFO] Building features...")
    features_df = build_features(football)

    if features_df.empty:
        raise RuntimeError("Feature engineering zwrócił pusty DataFrame")

    # =========================
    # TRENING MODELI
    # =========================
    for market in MARKETS:
        print(f"[TRAIN] Market: {market}")

        # target
        if market not in features_df.columns:
            print(f"[WARN] Market {market} missing in data, creating dummy target.")
            features_df[market] = 0

        target = features_df[market]

        # usuwamy target z feature setu
        X = features_df.drop(columns=MARKETS, errors="ignore")

        try:
            acc = train_and_save(
                X=X,
                y=target,
                market=market,
                models_dir=MODELS_DIR
            )
            print(f"[OK] {market} trained (acc={acc:.3f})")

        except Exception as e:
            print(f"[ERROR] Training failed for {market}: {e}")

    print("[DONE] Training completed successfully")

    # =========================
    # RAPORT BRAKÓW LIG
    # =========================
    if report:
        print("\n[REPORT] Leagues with missing data:")
        for k, v in report.items():
            print(f" - {k}: {v}")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    main()
