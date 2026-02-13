from data_loader import get_next_matches
from predictor import predict_matches

LEAGUES = {
    "EPL": 4328,
    "Bundesliga": 4331,
    "MLS": 4346
}

all_predictions = []

for name, league_id in LEAGUES.items():
    matches = get_next_matches(league_id)
    predictions = predict_matches(matches)
    predictions["League"] = name
    all_predictions.append(predictions)

import pandas as pd
final_df = pd.concat(all_predictions)

final_df.to_csv("predictions.csv", index=False)

print("Predictions generated.")
