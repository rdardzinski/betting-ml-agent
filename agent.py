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
    df_sorted = df.sort_values("ValueScore", ascending=False)
    coupons = []
    indices = df_sorted.index.tolist()
    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        coupons.append(indices[start:end])
    return coupons

# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================
def run():
    football = get_next_matches()
    if football.empty:
        print("[ERROR] Brak danych piłki nożnej.")
        return

    football = build_features(football)
    football_pred = predict(football)

    # Ustawienie ValueScore i Market (wybór rynku do kuponów)
    # Możesz zmienić 'Over25' na dowolny inny rynek lub logikę
    football_pred["ValueScore"] = football_pred["Over25_Prob"]
    football_pred["Market"] = "Over25"

    # --- ZAPIS ---
    football_pred.to_csv("predictions.csv", index=False)
    coupons = generate_coupons(football_pred)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"[INFO] Predictions saved: {len(football_pred)} rows")
    print(f"[INFO] Coupons saved: {len(coupons)}")
    print("[INFO] Agent finished successfully")

if __name__ == "__main__":
    run()
