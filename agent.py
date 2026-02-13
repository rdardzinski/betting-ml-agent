from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict
import pandas as pd
import os
import json

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

    # update metrics log
    metrics = {
        "num_predictions": len(predictions),
        "value_bets": predictions["ValueFlag"].sum()
    }
    with open("predictions_log.json","w") as f:
        json.dump(metrics,f)

    print("Predictions saved.")

if __name__ == "__main__":
    run()
