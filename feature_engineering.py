# feature_engineering.py
import pandas as pd

def build_features(df):
    # placeholder – przygotowuje cechy dla modeli
    df = df.copy()
    df['HomeForm'] = 1.0  # przykładowa cecha
    df['AwayForm'] = 1.0
    df['HomeRollingConceded'] = 0
    df['AwayRollingConceded'] = 0
    df['ELO_Home'] = 1500
    df['ELO_Away'] = 1500
    df['Odds_Home'] = 1.5
    df['Odds_Draw'] = 3.0
    df['Odds_Away'] = 2.5
    return df
