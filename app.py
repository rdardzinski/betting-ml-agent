import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# --- Wczytanie danych ---
df = pd.read_csv("predictions.csv")
with open("coupons.json") as f:
    coupons = json.load(f)

st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne rynki (Over25, BTTS, 1HGoals, 2HGoals, Cards, Corners)
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# --- Panele z kuponami ---
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
        for idx in coupons[i]:
            row = df.loc[idx]

            st.markdown(f"📅 **{row['Date']}**")
            st.markdown(f"⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**")

            for market in ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]:
                prob_col = f"{market}_Prob"
                acc_col = f"{market}_ModelAccuracy"
                flag_col = f"{market}_ValueFlag"

                if prob_col in row:
                    st.markdown(
                        f"- {market}: {round(row[prob_col]*100,1)}% | "
                        f"Acc: {round(row[acc_col]*100,1)}% | "
                        f"ValueFlag: {'✅' if row[flag_col] else '❌'}"
                    )
        st.markdown("---")
