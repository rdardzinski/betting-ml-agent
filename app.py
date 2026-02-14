import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# --- Wczytanie danych ---
df = pd.read_csv("predictions.csv")
with open("coupons.json") as f:
    coupons = json.load(f)

# --- Legenda ---
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – wszystkie rynki: Over 2.5 gola, BTTS, 1HGoals, 2HGoals, Cards, Corners
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# --- Filtry ---
leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
league_filter = st.selectbox("Wybierz ligę", leagues)

if league_filter != "All":
    df_filtered = df[df["League"] == league_filter]
else:
    df_filtered = df.copy()

# --- Zakładki kuponów ---
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
        for idx in coupons[i]:
            if idx >= len(df_filtered):
                continue
            row = df_filtered.iloc[idx]
            st.markdown(
                f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                f"Liga: {row['League']}  \n"
                f"Typy i prawdopodobieństwa:  \n"
                f"Over25_Prob: {round(row.get('Over25_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('Over25_ValueFlag',False) else '❌'}  \n"
                f"BTTS_Prob: {round(row.get('BTTS_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('BTTS_ValueFlag',False) else '❌'}  \n"
                f"1HGoals_Prob: {round(row.get('1HGoals_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('1HGoals_ValueFlag',False) else '❌'}  \n"
                f"2HGoals_Prob: {round(row.get('2HGoals_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('2HGoals_ValueFlag',False) else '❌'}  \n"
                f"Cards_Prob: {round(row.get('Cards_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('Cards_ValueFlag',False) else '❌'}  \n"
                f"Corners_Prob: {round(row.get('Corners_Prob',0)*100,1)}%  | ValueFlag: {'✅' if row.get('Corners_ValueFlag',False) else '❌'}  \n"
                f"ValueScore: {round(row.get('ValueScore',0),2)}  \n"
            )
        st.markdown("---")
