import json
import pandas as pd
from data_loader import get_next_matches
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("Over25_Prob", ascending=False)
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])
    return coupons

# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================
def run():
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak danych piłkarskich!")
        return

    # --- Feature engineering ---
    football["HomeRollingGoals"] = football["FTHG"].rolling(5, min_periods=1).mean()
    football["AwayRollingGoals"] = football["FTAG"].rolling(5, min_periods=1).mean()

    # --- Predykcje ---
    football_pred, markets = predict(football)

    # --- Upewnij się, że kolumny z nazwami drużyn i ligi są obecne ---
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football[col]

    # --- Zapis wyników ---
    football_pred.to_csv("predictions.csv", index=False)

    # --- Generowanie kuponów ---
    coupons = generate_coupons(football_pred)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print("[INFO] Agent zakończył pracę. Predykcje zapisane w predictions.csv, kupony w coupons.json")

if __name__ == "__main__":
    run()
