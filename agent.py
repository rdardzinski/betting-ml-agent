from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict
import pandas as pd
import os
import json
import itertools

def generate_coupons(df, num_coupons=20, picks_per_coupon=4):
    """
    Generuje listę kuponów typu 3 z 4 z rankingowanych value bets.
    """
    coupons = []
    df_value = df.copy()

    # Ranking: najpierw Over25, potem BTTS
    df_value["Score"] = df_value.get("Over25_Prob",0) + df_value.get("BTTS_Prob",0)
    df_value = df_value.sort_values("Score", ascending=False)

    # Wybieramy top N meczów dla tworzenia kuponów
    top_matches = df_value.head(num_coupons * picks_per_coupon)
    matches_list = list(top_matches.index)

    # Tworzenie kombinacji 3 z 4
    for i in range(num_coupons):
        selected = matches_list[i*picks_per_coupon:(i+1)*picks_per_coupon]
        for combo in itertools.combinations(selected, picks_per_coupon-1):
            coupons.append(list(combo))

    return coupons

def run():
    # 1️⃣ Pobranie nadchodzących meczów
    matches = get_next_matches()
    if matches.empty:
        print("No matches found")
        return

    # 2️⃣ Budowa cech
    matches = build_features(matches)

    # 3️⃣ Predykcje multi-market
    predictions = predict(matches)

    # 4️⃣ Zachowanie kolumn drużyn i ligi
    for col in ["HomeTeam","AwayTeam","League","Date"]:
        if col not in predictions.columns and col in matches.columns:
            predictions[col] = matches[col]

    # 5️⃣ Zapis do CSV
    temp_file = "predictions_temp.csv"
    predictions.to_csv(temp_file,index=False)
    os.replace(temp_file,"predictions.csv")

    # 6️⃣ Generowanie kuponów
    coupons_raw = generate_coupons(predictions)
    with open("coupons.json","w") as f:
        json.dump(coupons_raw,f)

    # 7️⃣ Aktualizacja logów
    metrics = {
        "num_predictions": int(len(predictions)),
        "value_bets": int(predictions.get("Over25_ValueFlag",0).sum() + predictions.get("BTTS_ValueFlag",0).sum())
    }
    # konwersja do typów Python
    metrics = {k: (int(v) if hasattr(v,'__int__') else float(v)) for k,v in metrics.items()}

    with open("predictions_log.json","w") as f:
        json.dump(metrics,f)

    print("Predictions and coupons saved.")
    print(f"Number of generated coupons: {len(coupons_raw)}")

if __name__ == "__main__":
    run()
