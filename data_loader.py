import requests
import pandas as pd

def get_next_matches(league_id):
    url = f"https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id={league_id}"
    r = requests.get(url)
    data = r.json()

    matches = []

    for event in data["events"]:
        matches.append({
            "HomeTeam": event["strHomeTeam"],
            "AwayTeam": event["strAwayTeam"],
            "Date": event["dateEvent"]
        })

    return pd.DataFrame(matches)
