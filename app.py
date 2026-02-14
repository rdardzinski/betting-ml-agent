import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

df = pd.read_csv("predictions.csv")
with open("coupons.json") as f:
    coupons = json.load(f)

st.markdown("""
**Legenda:**
- ⚽ Piłka nożna – różne typy bukmacherskie
- `Prob` – przewidywane prawdopodobieństwo wyniku
- ValueFlag – ✅ wartościowy zakład (>55%)
- ModelAccuracy – dokładność modelu
""")
st.markdown("---")

tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")

        for idx in coupons[i]:
            row = df.loc[idx]
            date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d") if "Date" in row else "Unknown"
            st.markdown(f"⚽ {date_str}: **{row.get('HomeTeam','Unknown')} vs {row.get('AwayTeam','Unknown')}** | Liga: {row.get('League','Unknown')}")

            # wszystkie typy bukmacherskie w jednej linii
            types = []
            for market in ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]:
                if market+"_Prob" in row:
                    prob = round(row[market+"_Prob"]*100,1)
                    flag = "✅" if row[market+"_ValueFlag"] else "❌"
                    types.append(f"{market}: {prob}% {flag}")
                elif market in row:
                    types.append(f"{market}: {row[market]}")

            if types:
                st.markdown(" | ".join(types))
        st.markdown("---")
