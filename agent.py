import pandas as pd
import json
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

def generate_coupons(df, n_coupons=5, picks=5):
    df = df.sort_values("Over25_Prob", ascending=False)  # top value score
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        coupons.append(indices[i*picks:(i+1)*picks])
    return coupons

def run():
    # --- Load football ---
    football = get_next_matches()
    football = build_features(football)

    # --- Predict ---
    football_pred = predict(football)

    for col in ["HomeTeam","AwayTeam","League","Date"]:
        if col in football.columns:
            football_pred[col] = football[col]
        else:
            football_pred[col] = "Unknown"

    football_pred["Sport"] = "Football"
    football_pred["ValueScore"] = football_pred.get("Over25_Prob", 0)

    # --- Save predictions ---
    football_pred.to_csv("predictions.csv", index=False)

    # --- Generate coupons ---
    coupons = generate_coupons(football_pred)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print("Agent finished successfully")

if __name__ == "__main__":
    run()
