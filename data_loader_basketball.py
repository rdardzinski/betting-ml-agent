import requests
import pandas as pd
from datetime import datetime, timedelta

def get_basketball_games(pages=3):
    """
    Pobiera mecze NBA/Euroliga z 3 różnych API i CSV fallback
    Zwraca dataframe z kolumnami: Date, HomeTeam, AwayTeam, League, HomeScore, AwayScore
    """

    cutoff_date = datetime.today() - timedelta(days=180)
    nba_data = []

    apis = [
        "https://www.balldontlie.io/api/v1/games",
        # dodaj inne API jeśli dostępne np. europejska liga
        #"https://api.example.com/euroleague/games"
    ]

    # =========================
    # 1️⃣ Próba API
    # =========================
    for api_url in apis:
        for page in range(1, pages + 1):
            try:
                resp = requests.get(f"{api_url}?per_page=100&page={page}", timeout=10)
                if resp.status_code != 200:
                    print(f"[WARN] Strona {page} API {api_url} status_code={resp.status_code}")
                    continue
                data = resp.json().get("data", [])
                for g in data:
                    date = pd.to_datetime(g.get("date", g.get("game_date", None)), errors="coerce")
                    if date < cutoff_date:
                        continue
                    nba_data.append({
                        "Date": date,
                        "HomeTeam": g["home_team"]["full_name"] if "home_team" in g else g.get("home_team_name","Unknown"),
                        "AwayTeam": g["visitor_team"]["full_name"] if "visitor_team" in g else g.get("away_team_name","Unknown"),
                        "League": "NBA",
                        "HomeScore": g.get("home_team_score", g.get("home_score", 0)),
                        "AwayScore": g.get("visitor_team_score", g.get("away_score", 0))
                    })
            except Exception as e:
                print(f"[ERROR] Strona {page} API {api_url}: {e}")

    df = pd.DataFrame(nba_data)

    # =========================
    # 2️⃣ Fallback CSV jeśli brak danych
    # =========================
    if df.empty:
        print("[INFO] Brak danych z API, fallback do CSV")
        try:
            url_csv = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
            df_csv = pd.read_csv(url_csv)
            df_csv = df_csv[["date","home_team","visitor_team","home_points","visitor_points"]]
            df_csv.columns = ["Date","HomeTeam","AwayTeam","HomeScore","AwayScore"]
            df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors="coerce")
            df_csv["League"] = "NBA"
            df_csv = df_csv[df_csv["Date"] >= cutoff_date]
            df = df_csv
            print(f"[INFO] Fallback CSV pobrało {len(df)} meczów")
        except Exception as e:
            print(f"[ERROR] CSV fallback nie powiódł się: {e}")
            df = pd.DataFrame()

    return df
