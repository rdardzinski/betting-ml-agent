import streamlit as st
import pandas as pd
import os
from evaluate import evaluate

st.set_page_config(layout="wide")
st.title("📊 Free First Betting Agent Dashboard v3 – Multi-market")

if not os.path.exists("predictions.csv"):
    st.warning("No predictions yet. Run agent first.")
    st.stop()

df = pd.read_csv("predictions.csv")
tab1, tab2 = st.tabs(["🎯 Predykcje","📘 Legenda"])

with tab1:
    st.subheader("Nadchodzące mecze – Multi-market")
    st.dataframe(df,use_container_width=True)
    stats = evaluate()
    if stats:
        st.subheader("📈 Statystyki")
        st.json(stats)

with tab2:
    st.markdown("""
## 📘 Legenda

**Over25_Prob / BTTS_Prob** – prawdopodobieństwo danego rynku  
**Over25_Confidence / BTTS_Confidence** – pewność modelu w %  
**Over25_ValueFlag / BTTS_ValueFlag** – True jeśli >55%  
**Over25_ModelAccuracy / BTTS_ModelAccuracy** – skuteczność modelu na danych testowych  

System:
- Dane historyczne: Football-Data
- Nadchodzące mecze: TheSportsDB
- Model: RandomForestClassifier
- Aktualizacja: GitHub Actions
""")
