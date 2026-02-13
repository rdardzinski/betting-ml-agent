import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(layout="wide")
st.title("📊 Free First Betting Agent Dashboard v4 – Multi-market z drużynami")

# ======== Wczytanie predykcji ========
if not os.path.exists("predictions.csv"):
    st.warning("No predictions yet. Uruchom najpierw agent.py")
    st.stop()

df = pd.read_csv("predictions.csv")

# ======== Wczytanie kuponów i mapowanie na nazwy drużyn ========
if os.path.exists("coupons.json"):
    with open("coupons.json","r") as f:
        coupons_raw = json.load(f)
    # mapowanie indeksów na nazwy meczów
    coupons = []
    for coupon in coupons_raw:
        matches = [f"{df.loc[i,'HomeTeam']} vs {df.loc[i,'AwayTeam']}" for i in coupon]
        coupons.append(matches)
else:
    coupons = []

# ======== Tabs ========
tab1, tab2, tab3 = st.tabs(["🎯 Predykcje","📘 Legenda","💰 Kupony"])

with tab1:
    st.subheader("Nadchodzące mecze – Multi-market")
    # kolumny do pokazania
    display_cols = ["League","HomeTeam","AwayTeam","Date",
                    "Over25_Prob","Over25_Confidence","Over25_ValueFlag","Over25_ModelAccuracy",
                    "BTTS_Prob","BTTS_Confidence","BTTS_ValueFlag","BTTS_ModelAccuracy"]
    st.dataframe(df[display_cols], use_container_width=True)

    # Podsumowanie z JSON log
    if os.path.exists("predictions_log.json"):
        with open("predictions_log.json","r") as f:
            metrics = json.load(f)
        st.subheader("📈 Statystyki predykcji")
        st.json(metrics)

with tab2:
    st.markdown("""
## 📘 Legenda

**Over25_Prob / BTTS_Prob** – prawdopodobieństwo danego rynku  
**Over25_Confidence / BTTS_Confidence** – pewność modelu w %  
**Over25_ValueFlag / BTTS_ValueFlag** – True jeśli >55%  
**Over25_ModelAccuracy / BTTS_ModelAccuracy** – skuteczność modelu na danych testowych  

System:
- Dane historyczne: Football-Data / TheSportsDB
- Model: RandomForestClassifier
- Aktualizacja: GitHub Actions
""")

with tab3:
    st.subheader("💰 20 wygenerowanych kuponów typu 3 z 4")
    if coupons:
        for i, coupon in enumerate(coupons[:20]):
            st.markdown(f"**Kupon {i+1}:** {', '.join(coupon)}")
    else:
        st.info("Brak kuponów. Uruchom agent.py aby je wygenerować.")
