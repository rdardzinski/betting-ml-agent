import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football + Basketball")

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("predictions.csv")
    with open("coupons.json") as f:
        coupons = json.load(f)
except Exception as e:
    st.error(f"❌ Load error: {e}")
    st.stop()

if df.empty:
    st.warning("Brak danych predykcyjnych.")
    st.stop()

# =========================
# HELPERS
# =========================
def safe(row, col, default="—"):
    return row[col] if col in row and pd.notna(row[col]) else default


# =========================
# SIDEBAR – FILTRY
# =========================
st.sidebar.header("🎛 Filtry")

sports = ["All"] + sorted(df["Sport"].dropna().unique().tolist())
sport_filter = st.sidebar.selectbox("Sport", sports)

leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
league_filter = st.sidebar.selectbox("Liga", leagues)

filtered = df.copy()
if sport_filter != "All":
    filtered = filtered[filtered["Sport"] == sport_filter]
if league_filter != "All":
    filtered = filtered[filtered["League"] == league_filter]

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🎫 Kupony", "🔥 Top Value Bets", "🧾 Raw Data"])

# =========================
# TAB 1 – KUPONY
# =========================
with tab1:
    if not coupons:
        st.info("Brak kuponów.")
    else:
        coupon_tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

        for i, tab in enumerate(coupon_tabs):
            with tab:
                for idx in coupons[i]:
                    if idx >= len(df):
                        continue
                    row = df.iloc[idx]

                    home = safe(row, "HomeTeam")
                    away = safe(row, "AwayTeam")
                    league = safe(row, "League")
                    date = safe(row, "Date")

                    if row["Sport"] == "Football":
                        prob = float(row.get("Over25_Prob", 0))
                        st.markdown(
                            f"⚽ **{home} vs {away}**  \n"
                            f"Liga: `{league}` | Data: `{date}`  \n"
                            f"Typ: Over 2.5 gola ({round(prob*100,1)}%)"
                        )
                    else:
                        prob = float(row.get("HomeWin_Prob", 0))
                        st.markdown(
                            f"🏀 **{home} vs {away}**  \n"
                            f"Rozgrywki: `{league}` | Data: `{date}`  \n"
                            f"Typ: Zwycięstwo gospodarzy ({round(prob*100,1)}%)"
                        )

                st.markdown("---")

# =========================
# TAB 2 – TOP VALUE BETS
# =========================
with tab2:
    st.subheader("Top Value Bets")
    top = filtered.sort_values("ValueScore", ascending=False).head(20)

    st.dataframe(
        top[[
            "Sport","Date","League","HomeTeam","AwayTeam","ValueScore"
        ]],
        use_container_width=True
    )

# =========================
# TAB 3 – RAW DATA
# =========================
with tab3:
    st.subheader("Raw predictions.csv")
    st.dataframe(filtered, use_container_width=True)
