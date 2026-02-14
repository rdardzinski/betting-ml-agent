import pandas as pd

def build_features(df):
    """
    Buduje cechy dla modeli bukmacherskich piłki nożnej.
    Zakładamy, że df zawiera kolumny: Date, HomeTeam, AwayTeam, FTHG, FTAG
    Dodajemy m.in.:
    - rolling goals scored/conceded w 5 ostatnich meczach
    - forma zespołu (ostatnie 5 meczów)
    """
    df = df.copy()
    df.sort_values(["HomeTeam", "Date"], inplace=True)
    
    # Rolling Goals scored
    df["HomeRollingGoals"] = df.groupby("HomeTeam")["FTHG"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["AwayRollingGoals"] = df.groupby("AwayTeam")["FTAG"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Rolling Goals conceded
    df["HomeRollingConceded"] = df.groupby("HomeTeam")["FTAG"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df["AwayRollingConceded"] = df.groupby("AwayTeam")["FTHG"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Form: % zwycięstw w ostatnich 5 meczach (dummy: wygrana = gola >0)
    df["HomeForm"] = df.groupby("HomeTeam")["FTHG"].apply(lambda x: x.rolling(5, min_periods=1).apply(lambda s: (s>0).mean())).reset_index(0, drop=True)
    df["AwayForm"] = df.groupby("AwayTeam")["FTAG"].apply(lambda x: x.rolling(5, min_periods=1).apply(lambda s: (s>0).mean())).reset_index(0, drop=True)
    
    df.fillna(0, inplace=True)
    return df
