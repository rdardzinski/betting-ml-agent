import json
import pandas as pd
from pathlib import Path

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================

def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("ValueScore", ascending=False).reset_index(drop=True)
    coupons = []

    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        coupons.append(list(range(start, min(end, len(df)))))

    return coupons


# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================

def run():
    print("[INFO] Loading football matches...")
    football = get_next_matches()
    print(f"[INFO] Football matches loaded: {len(football)}")

    if football.empty:
        raise RuntimeError("Brak danych piłkarskich")

    football = build_features(football)

    print("[INFO] Running football predictions...")
    football_pred, markets = predict(football)

    # Kolumny opisowe – ZAWSZE zachowane
    for col in ["Date", "HomeTeam", "AwayTeam", "League"]:
        football_pred[col] = football[col].values

    football_pred["Sport"] = "Football"

    # =========================
    # VALUE SCORE (max z rynków)
    # =========================

    prob_cols = [c for c in football_pred.columns if c.endswith("_Prob")]
    football_pred["ValueScore"] = football_pred[prob_cols].max(axis=1)

    # =========================
    # (Koszykówka – WYŁĄCZONA)
    # =========================
    """
    from data_loader_basketball import get_basketball_games
    basketball = get_basketball_games()
    ...
    """

    combined = football_pred.copy()

    # =========================
    # TOP 30% VALUE
    # =========================

    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold].reset_index(drop=True)

    # =========================
    # ZAPIS
    # =========================

    combined.to_csv("predictions.csv", index=False)

    coupons = generate_coupons(combined)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f, indent=2)

    print(f"[INFO] Predictions saved: {len(combined)} rows")
    print(f"[INFO] Coupons saved: {len(coupons)}")
    print("[INFO] Agent finished successfully")


if __name__ == "__main__":
    run()
