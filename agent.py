import os, joblib, json
import pandas as pd

MODELS = "models/football"
rows, coupons = [], []

df = pd.read_csv("data.csv")

for league in os.listdir(MODELS):
    for market in os.listdir(f"{MODELS}/{league}"):
        model_file = sorted(os.listdir(f"{MODELS}/{league}/{market}"))[-1]
        model = joblib.load(f"{MODELS}/{league}/{market}/{model_file}")

        subset = df[df["League"] == league]
        X = subset.select_dtypes("number")
        probs = model.predict_proba(X)[:,1]

        for i, p in zip(subset.index, probs):
            rows.append({
                "Date": subset.loc[i,"Date"],
                "League": league,
                "HomeTeam": subset.loc[i,"HomeTeam"],
                "AwayTeam": subset.loc[i,"AwayTeam"],
                "Market": market,
                "Prob": p
            })

pred = pd.DataFrame(rows)
pred.to_csv("predictions.csv", index=False)

for _, g in pred[pred["Prob"]>0.6].groupby("Date"):
    coupon = g.sort_values("Prob", ascending=False).head(5).index.tolist()
    coupons.append(coupon)

json.dump(coupons, open("coupons.json","w"))
