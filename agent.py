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
    football = build_features(football)
    football_pred = predict(football)

    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football[col]

    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred["Over25_Prob"]

    # --- KOSZYKÓWKA ---
    basketball = get_basketball_games()
    basketball["HomeWin"] = (basketball["HomeScore"] > basketball["AwayScore"]).astype(int)
    basketball["HomeWin_Prob"] = 0.55  # proxy (free-first)
    basketball["ValueScore"] = basketball["HomeWin_Prob"]
    basketball["Sport"] = "Basketball"

    basketball_pred = basketball[[
        "Date","HomeTeam","AwayTeam","League","Sport",
        "HomeWin_Prob","ValueScore"
    ]]

    # --- ŁĄCZENIE ---
    football_pred["HomeWin_Prob"] = None
    combined = pd.concat([football_pred, basketball_pred], ignore_index=True)

    # --- TOP 30% ---
    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold]

    # --- ZAPIS ---
    combined.to_csv("predictions.csv", index=False)

    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)

    print("Agent finished successfully")


if __name__ == "__main__":
    run()
