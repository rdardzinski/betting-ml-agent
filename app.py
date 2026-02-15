# app.py
import streamlit as st
from pathlib import Path
import json
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Betting ML Agent – Football")

COUPONS_FILE = "coupons.json"
ARCHIVE_DIR = Path("coupons_archive")
ARCHIVE_DIR.mkdir(exist_ok=True)

def load_coupons(file_path):
    if not Path(file_path).exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_weekend(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()
    saturday = today + timedelta((5 - today.weekday()) % 7)
    sunday = saturday + timedelta(1)
    next_saturday = saturday + timedelta(7)
    next_sunday = sunday + timedelta(7)
    return saturday.date() <= date.date() <= sunday.date(), next_saturday.date() <= date.date() <= next_sunday.date()

coupons = load_coupons(COUPONS_FILE)

tabs = st.tabs(["Aktualny weekend", "Następny weekend", "Archiwum"])

# ------------------------
# Aktualny weekend
# ------------------------
with tabs[0]:
    st.header("Kupony na aktualny weekend")
    for i, coupon in enumerate(coupons):
        filtered_bets = [b for b in coupon if is_weekend(b['Date'])[0]]
        if not filtered_bets:
            continue
        st.subheader(f"Kupon {i+1} ({len(filtered_bets)} zakładów)")
        matches_seen = {}
        for bet in filtered_bets:
            key = bet["Match"]
            matches_seen.setdefault(key, [])
            if len(matches_seen[key]) >= 3:
                continue
            matches_seen[key].append(bet)
            st.markdown(
                f"📅 {bet['Date']} | {bet['League']} | **{bet['Match']}**  \n"
                f"Typ: {bet['Market']} ({round(bet['Probability']*100,1)}%)  \n"
                f"Model Accuracy: {round(bet['ModelAccuracy']*100,1)}%  \n"
                f"ValueFlag: {'✅' if bet['ValueFlag'] else '❌'}"
            )

# ------------------------
# Następny weekend
# ------------------------
with tabs[1]:
    st.header("Kupony na następny weekend")
    for i, coupon in enumerate(coupons):
        filtered_bets = [b for b in coupon if is_weekend(b['Date'])[1]]
        if not filtered_bets:
            continue
        st.subheader(f"Kupon {i+1} ({len(filtered_bets)} zakładów)")
        matches_seen = {}
        for bet in filtered_bets:
            key = bet["Match"]
            matches_seen.setdefault(key, [])
            if len(matches_seen[key]) >= 3:
                continue
            matches_seen[key].append(bet)
            st.markdown(
                f"📅 {bet['Date']} | {bet['League']} | **{bet['Match']}**  \n"
                f"Typ: {bet['Market']} ({round(bet['Probability']*100,1)}%)  \n"
                f"Model Accuracy: {round(bet['ModelAccuracy']*100,1)}%  \n"
                f"ValueFlag: {'✅' if bet['ValueFlag'] else '❌'}"
            )

# ------------------------
# Archiwum
# ------------------------
with tabs[2]:
    st.header("Archiwalne kupony")
    for archive_file in sorted(ARCHIVE_DIR.glob("coupons_*.json"), reverse=True):
        archived = load_coupons(archive_file)
        st.subheader(f"{archive_file.stem} – {len(archived)} kuponów")
