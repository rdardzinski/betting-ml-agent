import os
import json
import pandas as pd
from datetime import datetime, timedelta

from data_loader import load_football_data
from feature_engineering import build_features
from predictor import train_and_save

# =========================
# KONFIGURACJA
# =========================

CUTOFF_MONTHS = 6
CUTOFF_DATE = datetime.utcnow() - timedelta(days=30 * CUTOFF_MONTHS)

MARKETS = {
    "Over25": {
        "target_col": "FTHG_FTAG_SUM_GT_25"
    },
    "BTTS": {
        "target_col": "BTTS_YN"
    },
    "1HGoals": {
        "target_col": "HTHG_HTAG_SUM_GT_05"
    },
    "2HGoals": {
        "target_col": "2H_GOALS_GT_05"
    },
    "Cards": {
        "target_col": "TOTAL_CARDS_GT_35"
    },
    "Corners": {
        "target_col": "TOTAL_CORNERS_GT_95"
    },
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# MAIN
# =========================

def main():
    print("[INFO] Loading football data...")

    matches, missing_leagues = load_football_data()

    if matches.empty:
        raise RuntimeError("No football data loaded – training aborted")

    # cutoff
    matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
    matches = matches[matches["Date"] >= CUTOFF_DATE]

    print(f"[INFO] Matches after cutoff ({CUTOFF_MONTHS} months): {len(matches)}")

    # =========================
    # FEATURE ENGINEERING
    # =========================
    print("[INFO] Building features...")

    features = build_features(matches)

    if features.empty:
        raise RuntimeError("Feature engineering returned empty dataframe")

    # =========================
    # TRAIN PER MARKET & LEAGUE
    # =========================

    summary = []

    for market, cfg in MARKETS.items():
        target_col = cfg["target_col"]

        print(f"\n[TRAIN] Market: {market}")

        if target_col not in features.columns:
            print(f"[WARN] Target {target_col} missing – creating dummy target")
            features[target_col] = 0

        for league in sorted(features["League"].unique()):
            league_df = features[features["League"] == league].copy()

            if len(league_df) < 200:
                print(f"[SKIP] {market} | {league} – too few samples ({len(league_df)})")
                continue

            try:
                acc = train_and_save(
                    df=league_df,
                    target_col=target_col,
                    market=market,
                    league=league,
                )

                summary.append({
                    "Market": market,
                    "League": league,
                    "Samples": len(league_df),
                    "Accuracy": round(acc, 3),
                })

                print(f"[OK] {market} | {league} acc={acc:.3f}")

            except Exception as e:
                print(f"[ERROR] {market} | {league}: {e}")

    # =========================
    # ZAPIS PODSUMOWANIA
    # =========================

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(MODELS_DIR, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "cutoff_months": CUTOFF_MONTHS,
        "markets": list(MARKETS.keys()),
        "missing_leagues": missing_leagues,
    }

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[INFO] Training completed successfully")
    print(f"[INFO] Summary saved to {summary_path}")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    main()
