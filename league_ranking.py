import pandas as pd
import json

def build_league_ranking(df):
    ranking = []

    for league in df["League"].unique():
        sub = df[df["League"]==league]
        acc = (sub["Over25_Prob"] > 0.5).mean()
        ranking.append({
            "League": league,
            "AccuracyProxy": round(acc,3),
            "Samples": int(len(sub))
        })

    ranking = sorted(ranking, key=lambda x: x["AccuracyProxy"], reverse=True)

    with open("league_ranking.json","w") as f:
        json.dump(ranking,f,indent=2)
