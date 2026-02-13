def build_features(df):
    # Rolling stats – placeholder dla nadchodzących meczów
    df["FTHG"] = 1
    df["FTAG"] = 1
    df["HomeRollingGoals"] = 1
    df["AwayRollingGoals"] = 1
    return df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]
