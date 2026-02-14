import os
import json
import pandas as pd
from data_loader import get_next_matches
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    if df.empty:
        return []

    df = df.sort_values("Over25_Prob", ascending=False)  # lub ValueScore
    coupons = []

    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])

    return coupons

# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================
def run():
    # --- PIŁKA NOŻNA ---
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak danych piłki nożnej – predykcja pominięta")
        return

    try:
        football_pred, available_markets = predict(football)
    except Exception as e:
        print(f"[ERROR] Predykcja football: {e}")
        return

    # zabezpieczenie kolumn
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football[col] if col in football.columns else None

    # ValueScore na podstawie Over25_Prob (można rozszerzyć)
    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred["Over25_Prob"] if "Over25_Prob" in football_pred.columns else 0

    # --- ZAPIS PREDYKCJI ---
    football_pred.to_csv("predictions.csv", index=False)

    # --- GENEROWANIE KUPONÓW ---
    coupons = generate_coupons(football_pred)
    with open("coupons.json","w") as f:
        json.dump(coupons, f)

    print(f"Football matches: {len(football_pred)}")
    print(f"Predictions saved: {len(football_pred)} rows")
    print(f"Coupons saved: {len(coupons)}")
    print("Agent finished successfully")

if __name__ == "__main__":
    run()
