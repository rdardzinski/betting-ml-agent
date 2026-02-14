import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy cechy wejściowe dla modeli piłki nożnej.
    Obsługuje wszystkie rynki: Over25, BTTS, 1HGoals, 2HGoals, Cards, Corners.

    Wymaga kolumn:
    - Date, HomeTeam, AwayTeam, FTHG, FTAG, HTG, ATG, HC, AC, HCorner, ACorner
    """

    df_sorted = df.sort_values("Date").copy()

    # Rolling features i form
    df_sorted["HomeRollingGoals"] = 0.0
    df_sorted["AwayRollingGoals"] = 0.0
    df_sorted["HomeRollingConceded"] = 0.0
    df_sorted["AwayRollingConceded"] = 0.0
    df_sorted["HomeForm"] = 0.0
    df_sorted["AwayForm"] = 0.0

    teams = pd.concat([df_sorted["HomeTeam"], df_sorted["AwayTeam"]]).unique()

    for team in teams:
        # Mecze gospodarzy i gości
        team_home = df_sorted[df_sorted["HomeTeam"] == team]
        team_away = df_sorted[df_sorted["AwayTeam"] == team]

        # Rolling 5 meczów
        team_home_goals = team_home["FTHG"].rolling(5, min_periods=1).mean()
        team_away_goals = team_away["FTAG"].rolling(5, min_periods=1).mean()
        team_home_conceded = team_home["FTAG"].rolling(5, min_periods=1).mean()
        team_away_conceded = team_away["FTHG"].rolling(5, min_periods=1).mean()

        # Form: ostatnie 5 zwycięstw
        home_results = (team_home["FTHG"] > team_home["FTAG"]).rolling(5, min_periods=1).mean()
        away_results = (team_away["FTAG"] > team_away["FTHG"]).rolling(5, min_periods=1).mean()

        # Wstawienie do df
        df_sorted.loc[df_sorted["HomeTeam"] == team, "HomeRollingGoals"] = team_home_goals.values
        df_sorted.loc[df_sorted["AwayTeam"] == team, "AwayRollingGoals"] = team_away_goals.values
        df_sorted.loc[df_sorted["HomeTeam"] == team, "HomeRollingConceded"] = team_home_conceded.values
        df_sorted.loc[df_sorted["AwayTeam"] == team, "AwayRollingConceded"] = team_away_conceded.values
        df_sorted.loc[df_sorted["HomeTeam"] == team, "HomeForm"] = home_results.values
        df_sorted.loc[df_sorted["AwayTeam"] == team, "AwayForm"] = away_results.values

    # Dodatkowe cechy dla rynków
    # Over25
    df_sorted["Over25"] = ((df_sorted["FTHG"] + df_sorted["FTAG"]) > 2.5).astype(int)
    # BTTS
    df_sorted["BTTS"] = ((df_sorted["FTHG"] > 0) & (df_sorted["FTAG"] > 0)).astype(int)
    # 1HGoals / 2HGoals
    df_sorted["1HGoals"] = df_sorted.get("HTG", 0)
    df_sorted["2HGoals"] = df_sorted.get("FTG", df_sorted["FTHG"]) - df_sorted.get("HTG", 0)
    # Cards
    df_sorted["Cards"] = df_sorted.get("HC", 0) + df_sorted.get("AC", 0)
    # Corners
    df_sorted["Corners"] = df_sorted.get("HCorner", 0) + df_sorted.get("ACorner", 0)

    # Fillna dla wszystkich brakujących
    df_sorted.fillna(0.5, inplace=True)

    return df_sorted
