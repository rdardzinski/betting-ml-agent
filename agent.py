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
    # Piłka nożna
    football = get_next_matches()
    football = build_features(football)
    football_pred = predict(football)
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        football_pred[col] = football[col]
    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred.get("Over25_Prob", 0.5)

    # Koszykówka
    basketball = get_basketball_games()
    if not basketball.empty:
        basketball_pred = predict(basketball)
        basketball_pred["Sport"] = "Basketball"
        basketball_pred["ValueScore"] = basketball_pred.get("HomeWin_Prob", 0.5)
        basketball_pred = basketball_pred.rename(columns={"HomeScore":"FTHG","AwayScore":"FTAG"})
    else:
        basketball_pred = pd.DataFrame(columns=football_pred.columns)

    football_pred["HomeWin_Prob"] = football_pred.get("Over25_Prob", 0.5)
    combined = pd.concat([football_pred, basketball_pred], ignore_index=True)

    threshold = combined["ValueScore"].quantile(0.7)
    combined = combined[combined["ValueScore"] >= threshold]

    combined.to_csv("predictions.csv", index=False)

    coupons = generate_coupons(combined)
    with open("coupons.json","w") as f:
        json.dump(coupons, f)

    print("Agent finished successfully")

if __name__ == "__main__":
    run()
