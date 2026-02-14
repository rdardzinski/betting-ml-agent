import pandas as pd

def build_features(df):
    """
    Dodaje kolumny pomocnicze do predykcji dla piłki nożnej:
    - HomeRollingGoals / AwayRollingGoals: średnia goli w ostatnich 5 meczach
    - TotalGoals: suma goli w meczu
    """
    df = df.copy()
    df["TotalGoals"] = df["FTHG"] + df["FTAG"]

    # Rolling dla ostatnich 5 meczów
    df["HomeRollingGoals"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["AwayRollingGoals"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # Inne przykładowe feature'y dla bukmacherskich typów
    df["HomeWin"] = (df["FTHG"] > df["FTAG"]).astype(int)
    df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["Over25"] = (df["TotalGoals"] > 2.5).astype(int)

    # Jeżeli kolumny do predykcji są brakujące, wypełnij medianą
    for col in ["HomeRollingGoals","AwayRollingGoals"]:
        if col not in df.columns:
            df[col] = df[col].median() if not df.empty else 0

    return df
