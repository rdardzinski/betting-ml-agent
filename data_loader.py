import pandas as pd

# =========================
# PIŁKA NOŻNA – 20+ LIG (FREE)
# Źródło: football-data.co.uk
# =========================
FOOTBALL_LEAGUES = {
    # ENGLAND
    "ENG1": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "ENG2": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "ENG3": "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    # GERMANY
    "D1": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "D2": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
    # SPAIN
    "ES1": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "ES2": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    # ITALY
    "IT1": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "IT2": "https://www.football-data.co.uk/mmz4281/2526/I2.csv",
    # FRANCE
    "FR1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "FR2": "https://www.football-data.co.uk/mmz4281/2526/F2.csv",
    # NETHERLANDS
    "NL1": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
    # PORTUGAL
    "PT1": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
    # BELGIUM
    "BE1": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    # SCOTLAND
    "SC1": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    # TURKEY
    "TR1": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
    # GREECE
    "GR1": "https://www.football-data.co.uk/mmz4281/2526/G1.csv",
}

def get_next_matches():
    frames = []
    for league, url in FOOTBALL_LEAGUES.items():
        try:
            df = pd.read_csv(url)
            required_cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
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
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["HomeTeam","AwayTeam"])
    return data
