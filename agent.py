from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict
import pandas as pd
import os
import json
import itertools

def generate_coupons(df, num_coupons=20, picks_per_coupon=4):
    coupons = []
    df_value = df.copy()
    # prosty ranking według Over25_ValueFlag
    df_value = df_value.sort_values("Over25_ValueFlag", ascending=False)
    top_matches = df_value.head(num_coupons * picks_per_coupon)
    matches_list = list(top_matches.index)
    # 3 z 4
    for i in range(num_coupons):
        selected = matches_list[i*picks_per_coupon:(i+1)*picks_per_coupon]
        for combo in itertools.combinations(selected, picks_per_coupon-1):
            coupons.append(list(combo))
    return coupons

def run():
    matches = get_next_matches()
    if matches.empty:
        print("No matches found")
        return

    matches = build_features(matches)
    predictions = predict(matches)

    temp_file = "predictions_temp.csv"
    predictions.to_csv(temp_file,index=False)
    os.replace(temp_file,"predictions.csv")

    # generowanie kuponów
    coupons = generate_coupons(predictions)
    with open("coupons.json","w") as f:
        json.dump(coupons,f)

    # update metrics log
    metrics = {
        "num_predictions": len(predictions),
        "value_bets": predictions["Over25_ValueFlag"].sum() + predictions["BTTS_ValueFlag"].sum()
    }
    with open("predictions_log.json","w") as f:
        json.dump(metrics,f)

    print("Predictions and coupons saved.")

if __name__ == "__main__":
    run()
