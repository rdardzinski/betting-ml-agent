import json
import pandas as pd
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
from predictor import predict

def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("ValueScore", ascending=False)
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])
    return coupons

def run():
    # --- PIŁKA NOŻNA ---
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak meczów piłki nożnej")
        football_pred = pd.DataFrame()
    else:
        football = build_features(football)
        football_pred = predict(football)
        for col in ["HomeTeam","AwayTeam","League","Date"]:
            football_pred[col] = football[col]
        football_pred["Sport"] = "Football"
        # wybór najlepszego typu do ValueScore (przykładowo Over25)
        football_pred["ValueScore"] = football_pred["Over25_Prob"]

    # --- KOSZYKÓWKA ---
    basketball = get_basketball_games()
    if basketball.empty:
        print("[WARN] Brak meczów koszykówki")
        basketball_pred = pd.DataFrame()
    else:
        basketball["HomeWin"] = (basketball["HomeScore"] > basketball["AwayScore"]).astype(int)
        basketball["HomeWin_Prob"] = 0.55  # proxy free-first
        basketball["BasketPoints"] = basketball["HomeScore"]
        basketball["BasketSum"] = basketball["HomeScore"] + basketball["AwayScore"]
        basketball_pred = basketball[["Date","HomeTeam","AwayTeam","League","HomeWin_Prob","BasketPoints","BasketSum"]]
        basketball_pred["ValueScore"] = basketball_pred["BasketPoints"]
        basketball_pred["Sport"] = "Basketball"

    # --- ŁĄCZENIE ---
    football_pred_cols = ["HomeTeam","AwayTeam","League","Date","Over25_Prob","Over25_ModelAccuracy","ValueScore","Sport"]
    basketball_pred_cols = ["HomeTeam","AwayTeam","League","Date","HomeWin_Prob","BasketPoints","BasketSum","ValueScore","Sport"]

    combined = pd.DataFrame()
    if not football_pred.empty:
        combined = football_pred[football_pred_cols]
    if not basketball_pred.empty:
        combined = pd.concat([combined, basketball_pred[basketball_pred_cols]], ignore_index=True)

    # --- TOP 30% ---
    if not combined.empty:
        threshold = combined["ValueScore"].quantile(0.7)
        combined = combined[combined["ValueScore"] >= threshold]

    # --- ZAPIS ---
    combined.to_csv("predictions.csv", index=False)
    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)
    print(f"[INFO] Agent finished successfully: {len(combined)} predictions, {len(coupons)} coupons")

if __name__ == "__main__":
    run()
