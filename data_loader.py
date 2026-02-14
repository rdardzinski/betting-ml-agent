import pandas as pd

def get_next_matches():
    """
    Pobiera nadchodzące mecze piłkarskie
    """
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        df = pd.read_csv(url)
        return df[["HomeTeam","AwayTeam","League","Date"]].dropna()
    except Exception as e:
        print(f"Nie udało się pobrać danych piłki nożnej: {e}")
        return pd.DataFrame(columns=["HomeTeam","AwayTeam","League","Date"])
