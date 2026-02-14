import pandas as pd
from datetime import datetime, timedelta
import requests

# =========================
# PIŁKA NOŻNA – 20+ LIG, SEZON 25/26
# =========================
FOOTBALL_LEAGUES = {
    "ENG1": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "ENG2": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "ENG3": "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    "D1": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "D2": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
    "ES1": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "ES2": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    "IT1": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "IT2": "https://www.football-data.co.uk/mmz4281/2526/I2.csv",
    "FR1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "FR2": "https://www.football-data.co.uk/mmz4281/2526/F2.csv",
    "NL1": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
    "PT1": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
    "BE1": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    "SC1": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "TR1": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
    "GR1": "https://www.football-data.co.uk/mmz4281/2526/G1.csv",
}

# Typy bukmacherskie
DEFAULT_MARKETS = [
    "Over05", "Over15", "Over25", "BTTS",
    "HomeScore_Half1", "AwayScore_Half1",
    "HomeScore_Half2", "AwayScore_Half2",
    "HomeCards", "AwayCards",
    "HomeCorners", "AwayCorners",
    "HomeGoal", "AwayGoal",
    "TotalGoals"
]

def get_next_matches():
    frames = []
    cutoff_date = datetime.today() - pd.Timedelta(days=180)  # ostatnie 6 miesięcy

    for league, url in FOOTBALL_LEAGUES.items():
        try:
            df = pd.read_csv(url)
            required_cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
            if not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols].copy()
            df["League"] = league
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[df["Date"] >= cutoff_date]

            frames.append(df)

        except Exception as e:
            print(f"[WARN] {league} not loaded: {e}")

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["HomeTeam","AwayTeam"])
    return data
