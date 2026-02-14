import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football (Full Markets)")

# =========================
# Wczytanie danych
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
# Filtry i legenda
# =========================
leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
selected_league = st.selectbox("Filtruj po lidze", leagues)

st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – wszystkie typy zakładów
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – True = wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
- Typy zakładów dostępne w aplikacji: 
    - Over 2.5 gola
    - BTTS (obie drużyny strzelą)
    - 1HGoals (gole w 1 połowie)
    - 2HGoals (gole w 2 połowie)
    - Cards (liczba kartek)
    - Corners (liczba rzutów rożnych)
""")
st.markdown("---")

# =========================
# Filtrowanie danych
# =========================
if selected_league != "All":
    df = df[df["League"] == selected_league]

# =========================
# Funkcja pomocnicza do wyświetlania zakładów
# =========================
def display_bet(row, market_name, icon="⚽"):
    prob_col = f"{market_name}_Prob"
    flag_col = f"{market_name}_ValueFlag"
    acc_col = f"{market_name}_ModelAccuracy"

    if prob_col in row and flag_col in row and acc_col in row:
        st.markdown(
            f"{icon} **{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
            f"Liga: {row['League']}  \n"
            f"Typ: {market_name} ({round(row[prob_col]*100,1)}%)  \n"
            f"Model Accuracy: {round(row[acc_col]*100,1)}%  \n"
            f"ValueFlag: {'✅' if row[flag_col] else '❌'}"
        )

# =========================
# Wyświetlanie kuponów
# =========================
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
        for idx in coupons[i]:
            if idx >= len(df):
                continue
            row = df.iloc[idx]

            # Wyświetlamy wszystkie typy bukmacherskie dla danego meczu
            for market in ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]:
                display_bet(row, market)
        st.markdown("---")
