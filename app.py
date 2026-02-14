import streamlit as st
import pandas as pd
import json

# =========================
# Streamlit config
# =========================
st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football + Basketball 25/26")

# =========================
# Load predictions & coupons
# =========================
try:
    df = pd.read_csv("predictions.csv")
except FileNotFoundError:
    st.error("Brak predictions.csv – uruchom najpierw agenta")
    st.stop()

try:
    with open("coupons.json") as f:
        coupons = json.load(f)
except FileNotFoundError:
    st.error("Brak coupons.json – uruchom najpierw agenta")
    st.stop()

# =========================
# LEGEND
# =========================
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne typy: Over 0.5, 1.5, 2.5, BTTS, Gole w połowie, Kartki, Rzuty rożne
- 🏀 Koszykówka – Zwycięstwo gospodarzy, Punkty Home/Away, TotalPoints
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – ✅ = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# =========================
# Tabs – kupony
# =========================
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")

        for idx in coupons[i]:
            row = df.loc[idx]

            # FOOTBALL
            if row["Sport"] == "Football":
                markets = [c for c in df.columns if "_Prob" in c and c not in ["HomeWin_Prob","HomeScore_Prob","AwayScore_Prob","TotalPoints_Prob"]]
                st.markdown(f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  \nLiga: {row['League']}")
                for m in markets:
                    prob = round(row.get(m,0)*100,1)
                    val_flag = '✅' if row.get(m,0) > 0.55 else '❌'
                    acc = round(row.get(m.replace("_Prob","_ModelAccuracy"),0)*100,1)
                    st.markdown(f"- Typ: {m.replace('_Prob','')} ({prob}%)  ModelAcc: {acc}%  ValueFlag: {val_flag}")

            # BASKETBALL
            else:
                markets = [c for c in df.columns if "_Prob" in c and c in ["HomeWin_Prob","HomeScore_Prob","AwayScore_Prob","TotalPoints_Prob"]]
                st.markdown(f"🏀 **{row['HomeTeam']} vs {row['AwayTeam']}**  \nRozgrywki: {row['League']}")
                for m in markets:
                    prob = round(row.get(m,0)*100,1)
                    val_flag = '✅' if row.get(m,0) > 0.55 else '❌'
                    acc = round(row.get(m.replace("_Prob","_ModelAccuracy"),0)*100,1)
                    st.markdown(f"- Typ: {m.replace('_Prob','')} ({prob}%)  ModelAcc: {acc}%  ValueFlag: {val_flag}")

        st.markdown("---")
