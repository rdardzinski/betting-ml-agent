import os
import json
import pandas as pd
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# =========================
# Generowanie kuponów
# =========================

def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("ValueScore", ascending=False)
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])
    return coupons

# =========================
# Główna logika agenta
# =========================

def run():
    # --- Piłka nożna ---
    football = get_next_matches()
    if football.empty:
        print("[WARN] No football matches found!")
        return

    football = build_features(football)

    try:
        football_pred = predict(football)
    except Exception as e:
        print(f"[ERROR] Football prediction failed: {e}")
        return

    # Uzupełnij brakujące kolumny drużyn i ligi
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        if col not in football_pred.columns:
            football_pred[col] = "Unknown"
        else:
            football_pred[col] = football_pred[col].fillna("Unknown")

    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred.get("Over25_Prob", pd.Series(0, index=football_pred.index))

    # --- Łączenie ---
    combined = football_pred.copy()

    # --- TOP 30% ---
    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold]

    # --- Zapis ---
    combined.to_csv("predictions.csv", index=False)
    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)

    print("Agent finished successfully")

if __name__ == "__main__":
    run()
