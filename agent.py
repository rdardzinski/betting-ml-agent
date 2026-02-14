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
        start = i*picks
        end = start+picks
        if start >= len(indices):
            break
        coupons.append(indices[start:end])
    return coupons

def run():
    # --- FOOTBALL ---
    football = get_next_matches()
    football_pred = predict(build_features(football)) if not football.empty else pd.DataFrame()

    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football.get(col, "Unknown")
    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred.get("Over25_Prob",0.5)

    # --- BASKETBALL ---
    basketball = get_basketball_games()
    basketball_pred = predict(basketball) if not basketball.empty else pd.DataFrame()
    if not basketball_pred.empty:
        basketball_pred["Sport"] = "Basketball"
        basketball_pred["ValueScore"] = basketball_pred.get("HomeWin_Prob",0.55)

    # --- CONCAT ---
    football_pred = football_pred.dropna(axis=1, how='all')
    basketball_pred = basketball_pred.dropna(axis=1, how='all')
    combined = pd.concat([football_pred, basketball_pred], ignore_index=True)

    combined.to_csv("predictions.csv", index=False)
    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)

    print(f"Predictions saved: {len(combined)} rows")
    print(f"Coupons saved: {len(coupons)}")
    print("Agent finished successfully")

if __name__=="__main__":
    run()
