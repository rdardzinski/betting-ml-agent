import streamlit as st
import pandas as pd
import json

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football (Multi-Market)")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("predictions.csv")

with open("coupons.json") as f:
    coupons = json.load(f)

# =========================
# VALIDATION
# =========================

required_cols = ["HomeTeam", "AwayTeam", "League", "Date", "Sport"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Brak wymaganych kolumn: {missing}")
    st.stop()

prob_cols = [c for c in df.columns if c.endswith("_Prob")]
acc_cols = [c for c in df.columns if c.endswith("_ModelAccuracy")]

# =========================
# FILTERS
# =========================

st.sidebar.header("Filtry")

leagues = ["All"] + sorted(df["League"].dropna().unique().tolist())
selected_league = st.sidebar.selectbox("Liga", leagues)

min_prob = st.sidebar.slider("Minimalne prawdopodobieństwo", 0.50, 0.90, 0.55)

if selected_league != "All":
    df = df[df["League"] == selected_league]

df = df[df[prob_cols].max(axis=1) >= min_prob]

# =========================
# LEGEND
# =========================

st.markdown("""
**Obsługiwane rynki (piłka nożna):**
- Over 2.5 gola
- BTTS
- Gole 1. połowa
- Gole 2. połowa
- Kartki
- Rzuty rożne
- Gole gospodarzy / gości

`ValueScore` = najwyższe prawdopodobieństwo z rynków
""")

st.markdown("---")

# =========================
# TABS = COUPONS
# =========================

tabs = st.tabs([f"Kupon {i+1}" for i in range(len(coupons))])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Kupon {i+1} ({len(coupons[i])} zakładów)")

        for idx in coupons[i]:
            if idx >= len(df):
                continue

            row = df.iloc[idx]

            st.markdown(
                f"""
⚽ **{row['HomeTeam']} vs {row['AwayTeam']}**  
Liga: {row['League']}  
Data: {row['Date']}
"""
            )

            for col in prob_cols:
                prob = row[col]
                if prob >= min_prob:
                    acc_col = col.replace("_Prob", "_ModelAccuracy")
                    acc = row[acc_col] if acc_col in df.columns else None

                    st.markdown(
                        f"- **{col.replace('_Prob','')}**: {round(prob*100,1)}% "
                        f"{'✅' if prob > 0.55 else '❌'} "
                        f"{f'(acc: {round(acc*100,1)}%)' if acc else ''}"
                    )

            st.markdown("---")
