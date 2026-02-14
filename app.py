import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# =========================
# Wczytaj dane
# =========================

try:
    df = pd.read_csv("predictions.csv")
except FileNotFoundError:
    st.error("Brak pliku predictions.csv. Uruchom najpierw agenta.")
    st.stop()

try:
    with open("coupons.json") as f:
        coupons = json.load(f)
except FileNotFoundError:
    st.error("Brak pliku coupons.json. Uruchom najpierw agenta.")
    st.stop()

# =========================
# Legenda
# =========================

st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne typy bukmacherskie
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
""")
st.markdown("---")

# =========================
# Filtr na ligę (opcjonalnie)
# =========================

if "League" in df.columns:
    leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
    selected_league = st.selectbox("Filtruj po lidze", leagues)
    if selected_league != "All":
        df = df[df["League"] == selected_league]

# =========================
# Tworzenie zakładek dla kuponów
# =========================

tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")

        displayed_matches = set()  # aby nie powielać typów tego samego meczu

        for idx in coupons[i]:
            if idx >= len(df):
                continue
            row = df.loc[idx]

            match_id = (row.get("HomeTeam", "Unknown"), row.get("AwayTeam", "Unknown"))
            if match_id in displayed_matches:
                continue  # pomiń powtórki
            displayed_matches.add(match_id)

            home = row.get("HomeTeam", "Unknown")
            away = row.get("AwayTeam", "Unknown")
            league = row.get("League", "Unknown")

            # Wyświetl wszystkie rynki
            markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]
            for market in markets:
                prob = row.get(f"{market}_Prob", None)
                acc = row.get(f"{market}_ModelAccuracy", None)
                if prob is None:
                    continue
                st.markdown(
                    f"⚽ **{home} vs {away}**  \n"
                    f"Liga: {league}  \n"
                    f"Typ: {market} ({round(prob*100,1)}%)  \n"
                    f"Model Accuracy: {round(acc*100,1) if acc is not None else 'N/A'}%  \n"
                    f"ValueFlag: {'✅' if prob>0.55 else '❌'}"
                )
        st.markdown("---")
