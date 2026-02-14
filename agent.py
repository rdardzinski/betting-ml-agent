import os
import json
import pandas as pd
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("Over25_Prob", ascending=False)  # sortowanie wg value
    coupons = []
    indices = list(df.index)

    for i in range(n_coupons):
        coupon_indices = indices[i*picks:(i+1)*picks]
        if coupon_indices:
            coupons.append(coupon_indices)
    return coupons

# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================
def run():
    print("[INFO] Loading football matches...")
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak danych piłki nożnej")
        return

    football = build_features(football)

    try:
        football_pred, markets = predict(football)
    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return

    for col in ["HomeTeam", "AwayTeam", "League", "Date"]:
        if col not in football_pred.columns:
            football_pred[col] = football[col]

    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred["Over25_Prob"]

    # =========================
    # TOP 30% wartościowe
    # =========================
    threshold = football_pred["ValueScore"].quantile(0.7)
    top_df = football_pred[football_pred["ValueScore"] >= threshold].reset_index(drop=True)

    # =========================
    # GENEROWANIE KUPONÓW
    # =========================
    coupons = generate_coupons(top_df, n_coupons=5, picks=5)

    # =========================
    # ZAPIS DO PLIKÓW
    # =========================
    top_df.to_csv("predictions.csv", index=False)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"[INFO] Agent finished successfully: {len(top_df)} predictions, {len(coupons)} coupons")

if __name__ == "__main__":
    run()
