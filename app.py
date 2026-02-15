import streamlit as st
import json
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

with open("coupons.json") as f:
    coupons = json.load(f)

with open("data_status.json") as f:
    missing = json.load(f)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏟 Aktualny weekend",
    "⏭ Następny weekend",
    "📦 Archiwum / ROI",
    "⚠️ Status danych"
])

def show_coupons(coupons):
    for i, c in enumerate(coupons):
        st.subheader(f"Kupon {i+1} | Confidence {c['Confidence']}%")
        for b in c["Bets"]:
            st.markdown(
                f"📅 {b['Date']} | {b['League']}  \n"
                f"**{b['Match']}**  \n"
                f"{b['Market']} | Prob {b['Probability']*100:.1f}% | "
                f"Value {'✅' if b['Value'] else '❌'}"
            )
        st.markdown("---")

with tab1:
    show_coupons(coupons)

with tab3:
    st.info("ROI i skuteczność będą liczone po zamknięciu zdarzeń")

with tab4:
    st.subheader("Brak danych dla lig:")
    for k, v in missing.items():
        st.write(f"- {v}")
