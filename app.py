import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football + Basketball")

df = pd.read_csv("predictions.csv")

with open("coupons.json") as f:
    coupons = json.load(f)

tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} (5 zakładów)")

        for idx in coupons[i]:
            row = df.loc[idx]

            if row["Sport"] == "Football":
                st.markdown(
                    f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                    f"Liga: {row['League']}  \n"
                    f"Typ: Over 2.5 gola ({round(row['Over25_Prob']*100,1)}%)"
                )
            else:
                st.markdown(
                    f"🏀 **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                    f"Rozgrywki: {row['League']}  \n"
                    f"Typ: Zwycięstwo gospodarzy ({round(row['HomeWin_Prob']*100,1)}%)"
                )
