import requests
import pandas as pd

def get_basketball_games(pages=3):
    """
    Pobiera mecze NBA z balldontlie.io API z paginacją.
    
    Args:
        pages (int): liczba stron do pobrania, każda strona max 100 meczów
    Returns:
        pd.DataFrame: kolumny Date, HomeTeam, AwayTeam, League, HomeScore, AwayScore
    """
    nba_data = []

    for page in range(1, pages + 1):
        url = f"https://www.balldontlie.io/api/v1/games?per_page=100&page={page}"
        try:
            resp = requests.get(url)
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
            else:
                print(f"[WARNING] Strona {page}: status_code={resp.status_code}")
        except Exception as e:
            print(f"[ERROR] Strona {page}: {e}")

    df = pd.DataFrame(nba_data)
    if df.empty:
        print("[INFO] Brak meczów NBA po paginacji.")
    else:
        print(f"[INFO] Pobranie {len(df)} meczów NBA z {pages} stron.")
    return df
