import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# Wczytanie danych
df = pd.read_csv("predictions.csv")
with open("coupons.json") as f:
    coupons = json.load(f)

st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne rynki bukmacherskie
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# Filtrowanie po lidze (opcjonalnie)
if "League" in df.columns:
    leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
    selected_league = st.selectbox("Filtruj po lidze", leagues)
    if selected_league != "All":
        df = df[df["League"] == selected_league]

# Tworzenie zakładek dla kuponów
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])
for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
        for idx in coupons[i]:
            row = df.loc[idx]
            st.markdown(
                f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                f"Liga: {row['League']}  \n"
                f"Typ: {row['Market']} ({round(row[f'{row['Market']}_Prob']*100,1)}%)  \n"
                f"Model Accuracy: {round(row[f'{row['Market']}_ModelAccuracy']*100,1)}%  \n"
                f"ValueFlag: {'✅' if row[f'{row['Market']}_ValueFlag'] else '❌'}"
            )
        st.markdown("---")
