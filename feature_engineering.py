def build_features(df):
    """
    Dodaje przykładowe kolumny rolling goals do football i placeholdery dla koszykówki
    """
    if "FTHG" in df.columns and "FTAG" in df.columns:
        df["HomeRollingGoals"] = df["FTHG"].rolling(5,min_periods=1).mean()
        df["AwayRollingGoals"] = df["FTAG"].rolling(5,min_periods=1).mean()
    else:
        df["HomeRollingGoals"] = 0
        df["AwayRollingGoals"] = 0
    return df
