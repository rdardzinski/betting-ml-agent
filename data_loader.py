import requests
import pandas as pd

LEAGUES = {
    "EPL": 4328,
    "Bundesliga": 4331,
    "MLS": 4346
}

def get_next_matches():
    all_matches = []

    for name, league_id in LEAGUES.items():
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id={league_id}"
        r = requests.get(url)

        if r.status_code != 200:
            continue

        data = r.json()

        if not data or not data.get("events"):
            continue

        for event in data["events"]:
            all_matches.append({
                "League": name,
                "HomeTeam": event["strHomeTeam"],
                "AwayTeam": event["strAwayTeam"],
                "Date": event["dateEvent"]
            })

    return pd.DataFrame(all_matches)
