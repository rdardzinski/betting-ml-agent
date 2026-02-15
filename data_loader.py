import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FOOTBALL_CSV = DATA_DIR / "football_matches.csv"

# =========================
# WSZYSTKIE LIGI (30+)
# =========================
LEAGUES = {
    # Polska
    "PL1": "Poland Ekstraklasa",
    "PL2": "Poland I Liga",
    # Czechy
    "CZ1": "Czech First League",
    # Rumunia
    "RO1": "Romania Liga 1",
    # Chorwacja
    "HR1": "Croatia HNL",
    # Bułgaria
    "BG1": "Bulgaria Parva Liga",
    # Dania
    "DK1": "Denmark Superliga",
    # Norwegia
    "NO1": "Norway Eliteserien",
    # Szwecja
    "SE1": "Sweden Allsvenskan",
    # Top 5 lig europejskich
    "ENG1": "English Premier League",
    "ENG2": "English Championship",
    "ESP1": "La Liga",
    "ESP2": "La Liga 2",
    "ITA1": "Serie A",
    "ITA2": "Serie B",
    "GER1": "Bundesliga",
    "GER2": "2. Bundesliga",
    "FRA1": "Ligue 1",
    "FRA2": "Ligue 2",
    # Inne popularne ligi europejskie
    "NL1": "Netherlands Eredivisie",
    "PT1": "Portugal Primeira Liga",
    "BE1": "Belgium Pro League",
    "AT1": "Austria Bundesliga",
    "CH1": "Switzerland Super League",
    "TR1": "Turkey Super Lig",
    "GR1": "Greece Super League",
    "RU1": "Russia Premier League",
    "UA1": "Ukraine Premier League",
    "RO2": "Romania Liga 2",
    "DK2": "Denmark 1st Division",
    "NO2": "Norway OBOS-ligaen",
    "SE2": "Sweden Superettan",
    "CZ2": "Czech National League",
    "HR2": "Croatia 2. HNL",
}

# =========================
# FUNKCJA ŁADOWANIA DANYCH
# =========================
def load_football_data(months_back: int = 6, incremental: bool = True) -> pd.DataFrame:
    """
    Pobiera dane piłkarskie i zapisuje do CSV.
    Args:
        months_back: ile miesięcy danych wstecz pobrać
        incremental: jeśli True, pobiera tylko nowe mecze

    Returns:
        pd.DataFrame z kolumnami:
        Date, HomeTeam, AwayTeam, League, HomeGoals, AwayGoals
    """

    cutoff_date = datetime.utcnow() - timedelta(days=30 * months_back)

    # Wczytanie istniejącego CSV jeśli istnieje
    if FOOTBALL_CSV.exists() and incremental:
        existing = pd.read_csv(FOOTBALL_CSV, parse_dates=["Date"])
        last_date = existing["Date"].max()
        fetch_from = max(last_date, cutoff_date)
        print(f"[INFO] Incremental mode. Fetching matches from {fetch_from.date()}")
    else:
        existing = None
        fetch_from = cutoff_date
        print(f"[INFO] Full fetch mode. Fetching matches from {fetch_from.date()}")

    all_rows = []
    missing_leagues = {}

    for code, league_name in LEAGUES.items():
        try:
            # =========================
            # PLACEHOLDER: podłącz swoje API / CSV / scraping
            # =========================
            # W tym przykładzie generujemy dummy dane
            dummy = pd.DataFrame({
                "Date": [datetime.utcnow()],
                "HomeTeam": [f"{league_name} Team A"],
                "AwayTeam": [f"{league_name} Team B"],
                "HomeGoals": [1],
                "AwayGoals": [0],
                "League": [league_name],
            })

            # Filtr na datę przyrostową
            dummy = dummy[dummy["Date"] >= fetch_from]
            all_rows.append(dummy)

        except Exception as e:
            print(f"[WARN] League {league_name} not loaded: {e}")
            missing_leagues[league_name] = str(e)

    if not all_rows:
        raise RuntimeError("No football data fetched from any league")

    new_data = pd.concat(all_rows, ignore_index=True)

    # Scalanie z istniejącym CSV jeśli istnieje
    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Date", "HomeTeam", "AwayTeam", "League"]
        )
    else:
        combined = new_data

    combined = combined.sort_values("Date")
    combined.to_csv(FOOTBALL_CSV, index=False)

    print(f"[INFO] Football matches saved: {len(combined)}")

    # Raport brakujących lig
    if missing_leagues:
        print("\n[REPORT] Leagues not loaded:")
        for league, reason in missing_leagues.items():
            print(f" - {league}: {reason}")

    return combined
