import pandas as pd
from datetime import datetime, timedelta
from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import train_and_save

CUTOFF = datetime.now() - timedelta(days=180)

df = get_next_matches()
df = df[df["Date"] >= CUTOFF]

df = build_features(df)

for league, g in df.groupby("League"):
    for market in ["Over25","BTTS"]:
        train_and_save(g, market, league)
