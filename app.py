import streamlit as st
import pandas as pd
import json
import os
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football (Inteligentne kupony)")

# =========================
# Wczytanie danych
# =========================
pred_file = "predictions.csv"
coupon_file = "coupons.json"

if os.path.exists(pred_file):
    df = pd.read_csv(pred_file)
else:
    st.warning("Brak pliku predictions.csv – uruchom agenta najpierw")
    df = pd.DataFrame()

# =========================
# Filtrowanie po lidze
# =========================
if not df.empty and "League" in df.columns:
    leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
    league_filter = st.selectbox("Wybierz ligę:", leagues)
    if league_filter != "All":
        df = df[df["League"] == league_filter]

# =========================
# Obliczanie ValueScore
# =========================
markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

def calculate_value_score(row):
    scores = []
    for m in markets:
        prob_col = f"{m}_Prob"
        flag_col = f"{m}_ValueFlag"
        if prob_col in row and flag_col in row and pd.notna(row[prob_col]):
            # ValueScore = prob * ValueFlag (True=1, False=0)
            scores.append(row[prob_col] if row[flag_col] else 0)
    if scores:
        return np.mean(scores)
    return 0

if not df.empty:
    df["ValueScore"] = df.apply(calculate_value_score, axis=1)

# =========================
# Generowanie kuponów (top 30% ValueScore)
# =========================
def generate_coupons(df, n_coupons=5, picks=5):
    df_sorted = df.sort_values("ValueScore", ascending=False)
    threshold = df_sorted["ValueScore"].quantile(0.7)
    top_df = df_sorted[df_sorted["ValueScore"] >= threshold].copy()
    indices = list(top_df.index)
    coupons = []

    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        if start >= len(indices):
            break
        coupons.append(indices[start:end])

    return coupons

coupons = generate_coupons(df)

# =========================
# Legenda
# =========================
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne rynki bukmacherskie
- `Prob` – przewidywane prawdopodobieństwo wyniku
- `ValueFlag` – ✅ wartościowy zakład (>55%)
- `ModelAccuracy` – dokładność modelu
- `ValueScore` – średnia wartość wszystkich rynków dla meczu
""")
st.markdown("---")

# =========================
# Wyświetlanie kuponów
# =========================
if coupons:
    tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
            for idx in coupons[i]:
                if idx >= len(df):
                    continue
                row = df.loc[idx]

                home = row.get("HomeTeam", "???")
                away = row.get("AwayTeam", "???")
                league = row.get("League", "???")
                value_score = round(row.get("ValueScore", 0)*100,1)

                st.markdown(f"⚽ **{home} vs {away}**  \nLiga: {league}  \nValueScore: {value_score}%")
                for m in markets:
                    prob_col = f"{m}_Prob"
                    acc_col = f"{m}_ModelAccuracy"
                    flag_col = f"{m}_ValueFlag"
                    if prob_col in row and pd.notna(row[prob_col]):
                        st.markdown(
                            f"Typ: {m}  \n"
                            f"Prawdopodobieństwo: {round(row[prob_col]*100,1)}%  \n"
                            f"Model Accuracy: {round(row.get(acc_col,0)*100,1)}%  \n"
                            f"ValueFlag: {'✅' if row.get(flag_col) else '❌'}"
                        )
                st.markdown("---")
else:
    st.info("Brak wygenerowanych kuponów do wyświetlenia")

# =========================
# Retention / historia ostatnich predykcji
# =========================
st.markdown("## 🕘 Historia predykcji")
history_files = sorted([f for f in os.listdir() if f.startswith("predictions_") and f.endswith(".csv")])
if history_files:
    st.write(f"Ostatnie zapisane pliki predykcji: {', '.join(history_files[-5:])}")
    for file in history_files[-5:]:
        st.write(f"- {file}")
else:
    st.info("Brak historii predykcji w katalogu")
