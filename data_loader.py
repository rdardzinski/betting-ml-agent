import pandas as pd
from datetime import datetime, timedelta

SUPPORTED_LEAGUES = {
    # TOP
    "ENG1": "England Premier League",
    "ESP1": "Spain La Liga",
    "ITA1": "Italy Serie A",
    "GER1": "Germany Bundesliga",
    "FRA1": "France Ligue 1",

    # EUROPA
    "POL1": "Poland Ekstraklasa",
    "POL2": "Poland I Liga",
    "CZE1": "Czech First League",
    "ROU1": "Romania Liga 1",
    "CRO1": "Croatia HNL",
    "BUL1": "Bulgaria First League",
    "DEN1": "Denmark Superliga",
    "NOR1": "Norway Eliteserien",
    "SWE1": "Sweden Allsvenskan",
    "AUT1": "Austria Bundesliga",
    "SUI1": "Switzerland Super League",
    "BEL1": "Belgium Pro League",
    "NED1": "Netherlands Eredivisie",
    "POR1": "Portugal Primeira Liga",
    "GRE1": "Greece Super League",
    "TUR1": "Turkey Super Lig",
    "SCO1": "Scotland Premiership",
    "IRL1": "Ireland Premier Division",
    "FIN1": "Finland Veikkausliiga",
    "ISL1": "Iceland Urvalsdeild",
    "SRB1": "Serbia SuperLiga",
    "SVK1": "Slovakia Super Liga",
    "HUN1": "Hungary NB I",
    "UKR1": "Ukraine Premier League",
    "CYP1": "Cyprus First Division",
}

REQUIRED_COLS = [
    "Date", "HomeTeam", "AwayTeam", "League",
    "Over25", "BTTS", "Corners", "Cards",
    "Odds_Over25", "Odds_BTTS"
]


def load_football_data(path="data/football_matches.csv", months_back=6):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    for c in missing_cols:
        df[c] = None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    cutoff = datetime.utcnow() - timedelta(days=30 * months_back)
    df = df[df["Date"] >= cutoff]

    df["HomeTeam"] = df["HomeTeam"].fillna("Unknown")
    df["AwayTeam"] = df["AwayTeam"].fillna("Unknown")
    df["League"] = df["League"].fillna("Unknown")

    # raport braków lig
    present = set(df["League"].unique())
    missing_leagues = {
        code: name for code, name in SUPPORTED_LEAGUES.items()
        if name not in present
    }

    df = df.reset_index(drop=True)

    return df, missing_leagues


def upcoming_matches(df, days=10):
    today = datetime.utcnow().date()
    end = today + timedelta(days=days)
    return df[(df["Date"].dt.date >= today) &
              (df["Date"].dt.date <= end)].copy()
