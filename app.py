import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football + Basketball")

# =========================
# Wczytywanie predykcji i kuponów
# =========================
if os.path.exists("predictions.csv"):
    df = pd.read_csv("predictions.csv")
else:
    st.warning("Brak pliku predictions.csv")
    df = pd.DataFrame()

if os.path.exists("coupons.json"):
    with open("coupons.json") as f:
        coupons = json.load(f)
else:
    st.warning("Brak pliku coupons.json")
    coupons = []

# =========================
# Legenda
# =========================
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – Over 2.5 gola
- 🏀 Koszykówka – Zwycięstwo gospodarzy
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# =========================
# Wyświetlanie kuponów w zakładkach
# =========================
if len(coupons) == 0 or df.empty:
    st.info("Brak kuponów lub danych do wyświetlenia")
else:
    tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
            for idx in coupons[i]:
                if idx >= len(df):
                    continue
                row = df.loc[idx]

                if row.get("Sport") == "Football":
                    over_prob = row.get("Over25_Prob", 0.5)
                    accuracy = row.get("Over25_ModelAccuracy", 0.5)
                    st.markdown(
                        f"⚽ **{row.get('HomeTeam','?')} vs {row.get('AwayTeam','?')}**  \n"
                        f"Liga: {row.get('League','?')}  \n"
                        f"Typ: Over 2.5 gola ({round(over_prob*100,1)}%)  \n"
                        f"Model Accuracy: {round(accuracy*100,1)}%  \n"
                        f"ValueFlag: {'✅' if over_prob>0.55 else '❌'}"
                    )
                elif row.get("Sport") == "Basketball":
                    home_prob = row.get("HomeWin_Prob", 0.55)
                    accuracy = row.get("HomeWin_ModelAccuracy", 0.5)
                    st.markdown(
                        f"🏀 **{row.get('HomeTeam','?')} vs {row.get('AwayTeam','?')}**  \n"
                        f"Rozgrywki: {row.get('League','?')}  \n"
                        f"Typ: Zwycięstwo gospodarzy ({round(home_prob*100,1)}%)  \n"
                        f"Model Accuracy: {round(accuracy*100,1)}%  \n"
                        f"ValueFlag: {'✅' if home_prob>0.55 else '❌'}"
                    )
            st.markdown("---")
