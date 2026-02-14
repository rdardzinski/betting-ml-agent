import requests
import pandas as pd

def get_basketball_games(pages=3):
    """
    Pobiera mecze NBA z balldontlie.io API z paginacją.
    Jeśli API zwraca błąd 404, fallback na CSV z GitHub.

    Args:
        pages (int): liczba stron do pobrania, każda strona max 100 meczów
    Returns:
        pd.DataFrame: kolumny Date, HomeTeam, AwayTeam, League, HomeScore, AwayScore
    """
    nba_data = []

    for page in range(1, pages + 1):
        url = f"https://www.balldontlie.io/api/v1/games?per_page=100&page={page}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for g in data:
                    nba_data.append({
                        "Date": g["date"][:10],
                        "HomeTeam": g["home_team"]["full_name"],
                        "AwayTeam": g["visitor_team"]["full_name"],
                        "League": "NBA",
                        "HomeScore": g["home_team_score"],
                        "AwayScore": g["visitor_team_score"],
                    })
            elif resp.status_code == 404:
                print(f"[WARN] Strona {page} nie istnieje (404). Kończę pobieranie.")
                break
            else:
                print(f"[WARNING] Strona {page}: status_code={resp.status_code}")
        except Exception as e:
            print(f"[ERROR] Strona {page}: {e}")

    df = pd.DataFrame(nba_data)
    if df.empty:
        print("[INFO] Brak meczów NBA z API, używam CSV fallback")
        try:
            url_csv = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
            df_csv = pd.read_csv(url_csv)
            df_csv = df_csv[["date","home_team","visitor_team","home_points","visitor_points"]]
            df_csv.columns = ["Date","HomeTeam","AwayTeam","HomeScore","AwayScore"]
            df_csv["League"] = "NBA"
            df_csv["Date"] = pd.to_datetime(df_csv["Date"], errors="coerce")
            df_csv = df_csv.dropna(subset=["HomeTeam","AwayTeam"])
            print(f"[INFO] Pobranie {len(df_csv)} meczów NBA z CSV fallback")
            return df_csv
        except Exception as e:
            print(f"[ERROR] Fallback CSV nie powiódł się: {e}")
            return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["HomeTeam","AwayTeam"])
    print(f"[INFO] Pobranie {len(df)} meczów NBA z API")
    return df
