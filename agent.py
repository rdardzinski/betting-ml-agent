import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# =========================
# GENEROWANIE KUPÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df_sorted = df.sort_values("ValueScore", ascending=False)
    threshold = df_sorted["ValueScore"].quantile(0.7)
    top_df = df_sorted[df_sorted["ValueScore"] >= threshold]
    indices = list(top_df.index)

    coupons = []
    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        if start >= len(indices):
            break
        coupons.append(indices[start:end])
    return coupons

# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================
def run():
    print("[INFO] Loading football data...")
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak meczów piłki nożnej")
        return

    football = build_features(football)

    try:
        # predykcja, predictor zwraca DataFrame z kolumnami wejściowymi zachowanymi
        football_pred, _ = predict(football)
    except Exception as e:
        print("[ERROR] Błąd predykcji football:", e)
        return

    # Ustawienie ValueFlag dla wszystkich rynków
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    for m in markets:
        prob_col = f"{m}_Prob"
        flag_col = f"{m}_ValueFlag"
        acc_col = f"{m}_ModelAccuracy"

        if prob_col in football_pred:
            football_pred[flag_col] = football_pred[prob_col] > 0.55
            if acc_col not in football_pred:
                football_pred[acc_col] = 0.8  # domyślna dokładność

    # Obliczenie ValueScore (średnia ValueFlag * prob dla rynków)
    def calculate_value_score(row):
        scores = []
        for m in markets:
            prob_col = f"{m}_Prob"
            flag_col = f"{m}_ValueFlag"
            if prob_col in row and flag_col in row and pd.notna(row[prob_col]):
                scores.append(row[prob_col] if row[flag_col] else 0)
        return np.mean(scores) if scores else 0

    football_pred["ValueScore"] = football_pred.apply(calculate_value_score, axis=1)
    football_pred["Sport"] = "Football"

    # --- KOSZYKÓWKA (zakomentowana na razie) ---
    """
    from data_loader_basketball import get_basketball_games
    basketball = get_basketball_games()
    if not basketball.empty:
        basketball['HomeWin_Prob'] = 0.55
        basketball['ValueScore'] = basketball['HomeWin_Prob']
        basketball['Sport'] = 'Basketball'
    else:
        basketball = pd.DataFrame()
    """

    # --- ŁĄCZENIE ---
    combined = football_pred.copy()
    # combined = pd.concat([football_pred, basketball], ignore_index=True)  # kosz komentowany

    # --- ZAPIS PREDYKCJI ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pred_filename = f"predictions_{timestamp}.csv"
    combined.to_csv(pred_filename, index=False)
    combined.to_csv("predictions.csv", index=False)  # aktualne predykcje
    print(f"[INFO] Predictions saved: {len(combined)} rows -> {pred_filename}")

    # --- GENEROWANIE KUPÓW ---
    coupons = generate_coupons(combined, n_coupons=5, picks=5)
    with open("coupons.json","w") as f:
        json.dump(coupons, f)
    print(f"[INFO] {len(coupons)} kupony zapisane -> coupons.json")

    print("[INFO] Agent finished successfully")

if __name__ == "__main__":
    run()
