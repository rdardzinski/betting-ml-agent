import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# =========================
# Wczytanie danych
# =========================
try:
    df = pd.read_csv("predictions.csv")
except FileNotFoundError:
    st.error("Brak pliku predictions.csv. Uruchom agenta najpierw.")
    st.stop()

try:
    with open("coupons.json") as f:
        coupons = json.load(f)
except FileNotFoundError:
    st.error("Brak pliku coupons.json. Uruchom agenta najpierw.")
    st.stop()

# =========================
# LEGEND
# =========================
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – Over 2.5 gola
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# =========================
# FILTRY
# =========================
leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
selected_league = st.selectbox("Wybierz ligę:", leagues)

if selected_league != "All":
    df = df[df["League"] == selected_league].reset_index(drop=True)

# =========================
# TABY KUPOW
# =========================
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")

        for idx in coupons[i]:
            if idx >= len(df):
                continue  # zabezpieczenie gdy coupon wychodzi poza zakres
            row = df.loc[idx]

            st.markdown(
                f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                f"Liga: {row['League']}  \n"
                f"Typ: Over 2.5 gola ({round(row['Over25_Prob']*100,1)}%)  \n"
                f"Model Accuracy: {round(row['Over25_ModelAccuracy']*100,1)}%  \n"
                f"ValueFlag: {'✅' if row['Over25_Prob']>0.55 else '❌'}"
            )
        st.markdown("---")
