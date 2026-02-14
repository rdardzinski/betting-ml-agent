import json
import pandas as pd
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
from predictor import predict

# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    """
    Generuje listę kuponów po n_coupons, każdy z picks zakładami
    """
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
    # -----------------------
    # 1️⃣ PIŁKA NOŻNA
    # -----------------------
    football = get_next_matches()
    football = build_features(football)
    football_pred = predict(football)

    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football[col]

    football_pred["Sport"] = "Football"

    # Używamy Over25_Prob do ValueScore
    football_pred["ValueScore"] = football_pred.get("Over25_Prob", 0.5)
    
    # -----------------------
    # 2️⃣ KOSZYKÓWKA
    # -----------------------
    basketball = get_basketball_games()
    if not basketball.empty:
        basketball = basketball.copy()
        basketball_pred = predict(basketball)  # używamy nowego predictor.py
        basketball_pred["Sport"] = "Basketball"
        basketball_pred["ValueScore"] = basketball_pred.get("HomeWin_Prob", 0.5)

        # Kolumny wspólne z piłką
        basketball_pred = basketball_pred.rename(columns={
            "HomeScore": "FTHG",
            "AwayScore": "FTAG"
        })
        basketball_pred["HomeWin_Prob"] = basketball_pred.get("HomeWin_Prob", 0.55)
    else:
        basketball_pred = pd.DataFrame(columns=football_pred.columns)

    # -----------------------
    # 3️⃣ ŁĄCZENIE
    # -----------------------
    football_pred["HomeWin_Prob"] = football_pred.get("Over25_Prob", 0.5)
    combined = pd.concat([football_pred, basketball_pred], ignore_index=True)

    # -----------------------
    # 4️⃣ TOP 30%
    # -----------------------
    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold]

    # -----------------------
    # 5️⃣ ZAPIS PREDYKCJI
    # -----------------------
    combined.to_csv("predictions.csv", index=False)

    # -----------------------
    # 6️⃣ GENEROWANIE KUPONÓW
    # -----------------------
    coupons = generate_coupons(combined)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print("Agent finished successfully")


if __name__ == "__main__":
    run()
