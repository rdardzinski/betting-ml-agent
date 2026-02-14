import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football + Basketball")

# =========================
# LOAD
# =========================
try:
    df = pd.read_csv("predictions.csv")
    with open("coupons.json") as f:
        coupons = json.load(f)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

if df.empty:
    st.warning("Brak danych.")
    st.stop()

# =========================
# SAFE ACCESS
# =========================
def col(name, default=None):
    return df[name] if name in df.columns else default

def val(row, key, default="—"):
    return row[key] if key in row and pd.notna(row[key]) else default


# =========================
# SIDEBAR
# =========================
st.sidebar.header("🎛 Filtry")

sports = ["All"] + sorted(col("Sport", pd.Series()).dropna().unique().tolist())
sport_filter = st.sidebar.selectbox("Sport", sports)

leagues = ["All"] + sorted(col("League", pd.Series()).dropna().unique().tolist())
league_filter = st.sidebar.selectbox("Liga", leagues)

filtered = df.copy()
if sport_filter != "All" and "Sport" in df.columns:
    filtered = filtered[filtered["Sport"] == sport_filter]
if league_filter != "All" and "League" in df.columns:
    filtered = filtered[filtered["League"] == league_filter]

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🎫 Kupony", "🔥 Value Bets", "🧾 Dane"])

# =========================
# TAB 1
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
                    r = df.iloc[idx]

                    home = val(r, "HomeTeam")
                    away = val(r, "AwayTeam")
                    league = val(r, "League")
                    date = val(r, "Date")

                    if val(r, "Sport") == "Football":
                        prob = float(val(r, "Over25_Prob", 0))
                        st.markdown(
                            f"⚽ **{home} vs {away}**  \n"
                            f"Liga: `{league}` | Data: `{date}`  \n"
                            f"Over 2.5 gola ({round(prob*100,1)}%)"
                        )
                    else:
                        prob = float(val(r, "HomeWin_Prob", 0))
                        st.markdown(
                            f"🏀 **{home} vs {away}**  \n"
                            f"Rozgrywki: `{league}` | Data: `{date}`  \n"
                            f"Zwycięstwo gospodarzy ({round(prob*100,1)}%)"
                        )

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Top Value Bets")
    if "ValueScore" in filtered.columns:
        st.dataframe(
            filtered.sort_values("ValueScore", ascending=False)
            .head(25),
            use_container_width=True
        )
    else:
        st.info("Brak ValueScore")

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Raw predictions.csv")
    st.dataframe(filtered, use_container_width=True)
