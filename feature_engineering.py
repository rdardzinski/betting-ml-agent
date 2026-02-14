import pandas as pd

def build_features(df):
    """
    Tworzy cechy do predykcji piłki nożnej:
    - Rolling goals
    - Forma drużyn
    - Możliwe dodatkowe cechy: kartki, rogi itp.
    """
    df = df.copy()

    # Rolling Goals
    df["HomeRollingGoals"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["AwayRollingGoals"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Forma drużyn
    df["HomeForm"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df["AwayForm"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.rolling(5, min_periods=1).sum())

    # Uzupełnij brakujące wartości
    for col in ["HomeRollingGoals","AwayRollingGoals","HomeForm","AwayForm"]:
        if col not in df.columns:
            df[col] = pd.Series(0, index=df.index)
        else:
            df[col] = df[col].fillna(0)

    return df
