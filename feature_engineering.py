import pandas as pd

ROLLING_WINDOW = 5

def build_features(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values("Date")

    # Rolling goals
    df["HomeRollingGoals"] = df.groupby("HomeTeam")["FTHG"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(0, drop=True)
    df["AwayRollingGoals"] = df.groupby("AwayTeam")["FTAG"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(0, drop=True)

    df["HomeRollingConceded"] = df.groupby("HomeTeam")["FTAG"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(0, drop=True)
    df["AwayRollingConceded"] = df.groupby("AwayTeam")["FTHG"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(0, drop=True)

    # Form (last N games win %)
    df["HomeForm"] = df.groupby("HomeTeam")["FTHG"].apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    df["AwayForm"] = df.groupby("AwayTeam")["FTAG"].apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())

    # Fill missing
    df.fillna(0, inplace=True)
    return df
