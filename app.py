import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("⚽ Football Betting ML Agent")

# =========================
# LOAD
# =========================
df = pd.read_csv("predictions.csv")
with open("coupons.json") as f:
    coupons = json.load(f)

if df.empty:
    st.warning("Brak danych")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Filtry")

leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
league = st.sidebar.selectbox("Liga", leagues)

filtered = df.copy()
if league != "All":
    filtered = filtered[filtered["League"] == league]

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🎫 Kupony", "📊 Wszystkie rynki"])

# =========================
# KUPONY
# =========================
with tab1:
    for i, coupon in enumerate(coupons):
        st.subheader(f"Kupon {i+1}")

        for idx in coupon:
            row = df.iloc[idx]

            st.markdown(
                f"**{row['HomeTeam']} vs {row['AwayTeam']}**  \n"
                f"Liga: `{row['League']}` | Data: `{row['Date']}`  \n"
                f"- Over 2.5: `{round(row['Over25']*100,1)}%`  \n"
                f"- BTTS: `{round(row['BTTS']*100,1)}%`  \n"
                f"- Over 1.5: `{round(row['Over15']*100,1)}%`"
            )

        st.markdown("---")

# =========================
# WSZYSTKIE RYNKI
# =========================
with tab2:
    st.dataframe(
        filtered[[
            "Date","League","HomeTeam","AwayTeam",
            "Over25","BTTS","Over15","Under35",
            "HomeTeamScore","AwayTeamScore","ValueScore"
        ]].sort_values("ValueScore", ascending=False),
        use_container_width=True
    )
