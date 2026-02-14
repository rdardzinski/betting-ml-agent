import pandas as pd

# =========================
# NBA (FREE)
# =========================

def load_nba():
    url = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
    df = pd.read_csv(url)

    df = df[["date","home_team","visitor_team","home_points","visitor_points"]]
    df.columns = ["Date","HomeTeam","AwayTeam","HomeScore","AwayScore"]
    df["League"] = "NBA"

    return df


# =========================
# EUROPA + MIĘDZYNARODOWE
# =========================

def load_europe_international():
    url = "https://raw.githubusercontent.com/sshleifer/nba_csv/master/international_games.csv"
    df = pd.read_csv(url)

    df = df.rename(columns={
        "home":"HomeTeam",
        "away":"AwayTeam",
        "home_score":"HomeScore",
        "away_score":"AwayScore",
        "competition":"League"
    })

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")

    return df[["Date","HomeTeam","AwayTeam","HomeScore","AwayScore","League"]]


# =========================
# PUBLIC API
# =========================

def get_basketball_games():
    frames = []

    try:
        frames.append(load_nba())
    except:
        pass

    try:
        frames.append(load_europe_international())
    except:
        pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HomeTeam","AwayTeam"])

    return df
