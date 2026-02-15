import pandas as pd

def compute_elo(df, base=1500, k=20):
    elo = {}
    ratings = []

    for _, r in df.sort_values("Date").iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)

        rh, ra = elo[h], elo[a]
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))

        if r["FTHG"] > r["FTAG"]:
            sh = 1
        elif r["FTHG"] < r["FTAG"]:
            sh = 0
        else:
            sh = 0.5

        elo[h] += k * (sh - eh)
        elo[a] += k * ((1 - sh) - (1 - eh))

        ratings.append((rh, ra))

    df["HomeElo"], df["AwayElo"] = zip(*ratings)
    return df
