# app.py
import streamlit as st
import json
from pathlib import Path

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

# ------------------------
# Wczytaj kupony
# ------------------------
COUPONS_FILE = "coupons.json"

if not Path(COUPONS_FILE).exists():
    st.error("Brak pliku coupons.json. Uruchom najpierw agenta.")
    st.stop()

with open(COUPONS_FILE, "r", encoding="utf-8") as f:
    coupons = json.load(f)

# ------------------------
# Legenda
# ------------------------
st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – wszystkie typy bukmacherskie
- `Probability` – przewidywane prawdopodobieństwo wyniku
- `ModelAccuracy` – dokładność modelu
- `ValueFlag` – True = wartościowy zakład (>55%)
""")
st.markdown("---")

# ------------------------
# Pokaż zakłady w zakładkach
# ------------------------
tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")
        for bet in coupons[i]:
            st.markdown(
                f"📅 {bet['Date']} | {bet['League']} | **{bet['Match']}**  \n"
                f"Typ: {bet['Market']} ({round(bet['Probability']*100,1)}%)  \n"
                f"Model Accuracy: {round(bet['ModelAccuracy']*100,1)}%  \n"
                f"ValueFlag: {'✅' if bet['ValueFlag'] else '❌'}"
            )
        st.markdown("---")

# ------------------------
# Informacja o liczbie kuponów
# ------------------------
st.info(f"Liczba wygenerowanych kuponów: {len(coupons)}")
