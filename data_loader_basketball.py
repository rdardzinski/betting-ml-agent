import requests
import pandas as pd

def get_basketball_games():
    """
    Pobiera ostatnie 100 meczów NBA z balldontlie.io
    """
    nba_data = []
    url = "https://www.balldontlie.io/api/v1/games?per_page=100"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()["data"]
            for g in data:
                nba_data.append({
                    "Date": g["date"][:10],
                    "HomeTeam": g["home_team"]["full_name"],
                    "AwayTeam": g["visitor_team"]["full_name"],
                    "League": "NBA",
                    "HomeScore": g["home_team_score"],
                    "AwayScore": g["visitor_team_score"],
                })
        return pd.DataFrame(nba_data)
    except Exception as e:
        print(f"Błąd pobierania danych NBA: {e}")
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","League","HomeScore","AwayScore"])
