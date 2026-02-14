import pandas as pd

# =========================
# PIŁKA NOŻNA – 20+ LIG
# Źródło: football-data.co.uk (FREE, bez API key)
# =========================

FOOTBALL_LEAGUES = {
    # ENGLAND (3 poziomy)
    "ENG1": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "ENG2": "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
    "ENG3": "https://www.football-data.co.uk/mmz4281/2324/E2.csv",

    # GERMANY (2 poziomy)
    "D1": "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "D2": "https://www.football-data.co.uk/mmz4281/2324/D2.csv",

    # SPAIN (2 poziomy)
    "ES1": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "ES2": "https://www.football-data.co.uk/mmz4281/2324/SP2.csv",

    # ITALY (2 poziomy)
    "IT1": "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    "IT2": "https://www.football-data.co.uk/mmz4281/2324/I2.csv",

    # FRANCE (2 poziomy)
    "FR1": "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
    "FR2": "https://www.football-data.co.uk/mmz4281/2324/F2.csv",

    # NETHERLANDS
    "NL1": "https://www.football-data.co.uk/mmz4281/2324/N1.csv",

    # PORTUGAL
    "PT1": "https://www.football-data.co.uk/mmz4281/2324/P1.csv",

    # BELGIUM
    "BE1": "https://www.football-data.co.uk/mmz4281/2324/B1.csv",

    # SCOTLAND
    "SC1": "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",

    # TURKEY
    "TR1": "https://www.football-data.co.uk/mmz4281/2324/T1.csv",

    # GREECE
    "GR1": "https://www.football-data.co.uk/mmz4281/2324/G1.csv",

    # BRAZIL
    "BR1": "https://www.football-data.co.uk/mmz4281/2324/BRA.csv",

    # ARGENTINA
    "ARG1": "https://www.football-data.co.uk/mmz4281/2324/ARG.csv",

    # MLS (USA)
    "MLS": "https://www.football-data.co.uk/mmz4281/2324/USA.csv",
}

# =========================
# FUNKCJA GŁÓWNA – PIŁKA
# =========================

def get_next_matches():
    frames = []

    for league, url in FOOTBALL_LEAGUES.items():
        try:
            df = pd.read_csv(url)

            required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
            if not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols].copy()
            df["League"] = league

            frames.append(df)

        except Exception as e:
            print(f"[WARN] {league} not loaded: {e}")

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Standaryzacja
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["HomeTeam", "AwayTeam"])

    return data


# =========================
# NBA – KOSZYKÓWKA (FREE)
# =========================

def get_nba_games():
    """
    Dane historyczne NBA – do modelu zwycięzcy
    Źródło: basketball-reference.com (scrape-ready CSV)
    """
    try:
        url = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
        df = pd.read_csv(url)

        df = df[["date", "home_team", "visitor_team", "home_points", "visitor_points"]]
        df.columns = ["Date", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore"]
        df["League"] = "NBA"

        return df

    except Exception as e:
        print("[WARN] NBA data not loaded:", e)
        return pd.DataFrame()
