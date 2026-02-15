import streamlit as st
import pandas as pd
import json

df = pd.read_csv("predictions.csv")
coupons = json.load(open("coupons.json"))

st.title("⚽ Betting ML – Football")

for i, c in enumerate(coupons):
    st.subheader(f"Kupon {i+1}")
    for idx in c:
        r = df.loc[idx]
        st.markdown(
            f"📅 {r['Date']}  \n"
            f"**{r['HomeTeam']} vs {r['AwayTeam']}** ({r['League']})  \n"
            f"Typ: **{r['Market']}**  \n"
            f"Prawdopodobieństwo: **{round(r['Prob']*100,1)}%**"
        )
