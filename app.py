import streamlit as st
import plotly.express as px
from agent_core import load_data, train_models, predict_markets
from portfolio import build_portfolio
from risk import monte_carlo_bankroll
from meta import rank_markets

st.set_page_config(layout="wide")
st.title("HEDGE FUND AI BETTING LAB")

if st.button("Train Models"):
    df = load_data()
    results = train_models(df)
    st.write("Model accuracy per market:", results)

df = load_data()

try:
    selections = predict_markets(df)
except:
    st.warning("Train models first.")
    st.stop()

st.subheader("Top Value Selections")
st.dataframe(selections.head(20))

ranking = rank_markets(selections)
st.subheader("Market Ranking by EV")
st.bar_chart(ranking)

mean_br, worst_case = monte_carlo_bankroll(selections.head(20))

st.subheader("Monte Carlo Bankroll Simulation")
st.write("Expected Bankroll:", round(mean_br,2))
st.write("5% Worst Case:", round(worst_case,2))

coupons = build_portfolio(selections)

st.subheader("Generated Coupons")
for i, c in enumerate(coupons[:5]):
    st.write(f"Coupon {i+1}")
    st.dataframe(c[['HomeTeam','AwayTeam','Market','Probability','EV']])

fig = px.histogram(selections, x="EV")
st.plotly_chart(fig, use_container_width=True)
