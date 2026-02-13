import streamlit as st
import pandas as pd
import numpy as np
import glob

from model import train_model
from regime import detect_regime
from confidence import calculate_confidence
from capital_allocator import allocate_capital

st.set_page_config(layout="wide")

st.title("IQ 2.0 – Multiliga / Multisport Betting Intelligence")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    files = glob.glob("data/*.csv")
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file))
    df = pd.concat(df_list, ignore_index=True)
    return df

df = load_data()

# ==============================
# BASIC FEATURES
# ==============================

df["TotalGoals"] = df["FTHG"] + df["FTAG"]
df["Over25"] = (df["TotalGoals"] > 2.5).astype(int)
df["BTTS"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
df["Under25"] = (df["TotalGoals"] <= 2.5).astype(int)

# ==============================
# TRAIN MODEL
# ==============================

model, acc = train_model(df)

st.sidebar.header("Model Info")
st.sidebar.write(f"Model Accuracy: {round(acc*100,2)}%")

# ==============================
# MARKET SIMULATION
# ==============================

df["Probability"] = model.predict_proba(
    df[["FTHG", "FTAG"]]
)[:, 1]

df["Odds"] = np.random.uniform(1.5, 2.2, len(df))
df["EV"] = (df["Probability"] * df["Odds"]) - 1

# stability proxy
df["stability_index"] = (
    df.groupby("League")["EV"]
    .transform(lambda x: x.mean() / (x.std() + 0.001))
)

df["Confidence"] = df.apply(calculate_confidence, axis=1)

# ==============================
# FILTER VALUE
# ==============================

value_df = df[
    (df["EV"] > 0.05) &
    (df["Probability"] > 0.55)
]

value_df = value_df.sort_values(
    by="Confidence",
    ascending=False
)

# ==============================
# DASHBOARD TABS
# ==============================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Top Value Picks",
     "🌍 League Regime",
     "💰 Capital Allocation",
     "📘 Legenda"]
)

# ==============================
# TAB 1 – VALUE PICKS
# ==============================

with tab1:

    st.subheader("Top 20 Value Selections")

    st.dataframe(
        value_df.head(20),
        use_container_width=True,
        column_config={
            "League": st.column_config.TextColumn(
                "League",
                help="Liga rozgrywkowa"
            ),
            "HomeTeam": st.column_config.TextColumn(
                "Home",
                help="Drużyna gospodarzy"
            ),
            "AwayTeam": st.column_config.TextColumn(
                "Away",
                help="Drużyna gości"
            ),
            "Probability": st.column_config.NumberColumn(
                "Model Probability",
                help="Prawdopodobieństwo wyliczone przez model ML"
            ),
            "Odds": st.column_config.NumberColumn(
                "Market Odds",
                help="Przyjęty kurs rynkowy (symulacja)"
            ),
            "EV": st.column_config.NumberColumn(
                "Expected Value",
                help="(Probability × Odds) − 1"
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence Score",
                help="Ocena 0–100 uwzględniająca EV, stabilność ligi i jakość predykcji"
            ),
            "stability_index": st.column_config.NumberColumn(
                "Stability Index",
                help="EV / odchylenie standardowe EV w lidze"
            )
        }
    )

# ==============================
# TAB 2 – REGIME DETECTION
# ==============================

with tab2:

    regime = detect_regime(df)

    st.subheader("League Regime Detection")

    st.dataframe(
        regime,
        use_container_width=True,
        column_config={
            "League": st.column_config.TextColumn(
                "League",
                help="Liga analizowana przez system"
            ),
            "TotalGoals": st.column_config.NumberColumn(
                "Avg Goals",
                help="Średnia liczba goli w lidze"
            ),
            "Regime": st.column_config.TextColumn(
                "Regime",
                help="OVER – liga ofensywna / UNDER – liga defensywna"
            )
        }
    )

# ==============================
# TAB 3 – CAPITAL ALLOCATION
# ==============================

with tab3:

    st.subheader("Dynamic Capital Allocation")

    allocations = allocate_capital(value_df)

    st.write("Proponowany podział bankrolla (na podstawie średniego EV lig):")
    st.json(allocations)

# ==============================
# TAB 4 – LEGENDA
# ==============================

with tab4:

    st.markdown("""
## 📘 Legenda IQ 2.0

**League** – Liga rozgrywkowa  

**Home / Away** – Drużyny w meczu  

**TotalGoals** – Suma bramek w meczu  

**Probability** – Prawdopodobieństwo z modelu ML  

**Odds** – Kurs rynkowy (symulacja)  

**EV (Expected Value)** –  
Wzór: (Probability × Odds) − 1  

• EV > 0 → potencjalnie value  
• EV < 0 → brak przewagi  

**Stability Index** –  
Średnie EV w lidze / odchylenie standardowe  

**Confidence (0–100)** –  
Złożony wskaźnik uwzględniający:  
- EV  
- stabilność ligi  
- siłę predykcji  

**Regime** –  
OVER → liga z wysoką średnią goli  
UNDER → liga z niską średnią goli  

---

IQ 2.0 to system pół-instytucjonalny z dynamiczną alokacją kapitału.
    """)

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.caption("IQ 2.0 – Quant Betting Lab")
