import os
import pandas as pd
import json
from predictor import predict
from data_loader import get_next_matches
from feature_engineering import build_features

# =========================
# Generowanie kuponów
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("ValueScore", ascending=False)
    coupons = []

    indices = df.index.tolist()
    for i in range(n_coupons):
        start = i*picks
        end = start+picks
        if start >= len(indices):
            break
        coupons.append(indices[start:end])

    return coupons

# =========================
# Główny agent
# =========================
def run():
    # --- PIŁKA NOŻNA ---
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak danych piłki nożnej.")
        return

    football = build_features(football)

    features_cols = [
        "FTHG","FTAG",
        "HomeRollingGoals","AwayRollingGoals",
        "HomeRollingConceded","AwayRollingConceded",
        "HomeForm","AwayForm"
    ]
    features = football[features_cols]

    preds = predict(features)
    football_pred = pd.concat([football.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    # Obliczenie ValueScore – najprostsza strategia: max prob z wszystkich rynków
    prob_cols = [c for c in preds.columns if c.endswith("_Prob")]
    football_pred["ValueScore"] = football_pred[prob_cols].max(axis=1)

    # --- Zapis wyników ---
    football_pred.to_csv("predictions.csv", index=False)

    coupons = generate_coupons(football_pred)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"[INFO] Predictions saved: {len(football_pred)} rows")
    print(f"[INFO] Coupons saved: {len(coupons)}")
    print("[INFO] Agent finished successfully.")

if __name__ == "__main__":
    run()
