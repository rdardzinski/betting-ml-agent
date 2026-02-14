import json
import pandas as pd
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("ValueScore", ascending=False)
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
        print("[ERROR] Brak meczów piłki nożnej.")
        return

    football = build_features(football)

    # Przewidywania dla wszystkich rynków
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
    try:
        football_pred, _ = predict(football, markets=markets)
    except Exception as e:
        print("[ERROR] Błąd predykcji football:", e)
        return

    # Zachowaj nazwy drużyn i ligi
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        if col in football.columns:
            football_pred[col] = football[col]
        else:
            football_pred[col] = "Unknown"

    football_pred["Sport"] = "Football"

    # ValueScore: najwyższe prawdopodobieństwo z Over25 jako przykładowe kryterium
    football_pred["ValueScore"] = football_pred["Over25_Prob"]

    # --- KOSZYKÓWKA (komentowana) ---
    # basketball = get_basketball_games()
    # basketball["HomeWin"] = (basketball["HomeScore"] > basketball["AwayScore"]).astype(int)
    # basketball["HomeWin_Prob"] = 0.55  # proxy (free-first)
    # basketball["ValueScore"] = basketball["HomeWin_Prob"]
    # basketball["Sport"] = "Basketball"
    # basketball_pred = basketball[[
    #     "Date","HomeTeam","AwayTeam","League","Sport",
    #     "HomeWin_Prob","ValueScore"
    # ]]

    # --- ŁĄCZENIE ---
    combined = football_pred  # + basketball_pred (jeżeli odkomentujesz)
    
    # --- TOP 30% ---
    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold]

    # --- ZAPIS ---
    combined.to_csv("predictions.csv", index=False)

    # Generowanie kuponów
    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons, f)

    print(f"Football matches: {len(football_pred)}")
    print(f"Predictions saved: {len(combined)} rows")
    print(f"Coupons saved: {len(coupons)}")
    print("Agent finished successfully")

if __name__ == "__main__":
    run()
