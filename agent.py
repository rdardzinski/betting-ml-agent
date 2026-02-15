import json
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from data_loader import load_football_data, get_upcoming_matches
from predictor import predict_markets

# === KONFIG ===
MAX_BETS_PER_MATCH = 3
MAX_BETS_PER_COUPON = 5
MIN_PROBABILITY = 0.55

COUPONS_FILE = "coupons.json"
ARCHIVE_DIR = Path("coupons_archive")
ARCHIVE_DIR.mkdir(exist_ok=True)


# === DATY WEEKENDÓW ===
def current_and_next_weekend():
    today = datetime.utcnow().date()
    friday = today + timedelta((4 - today.weekday()) % 7)
    sunday = friday + timedelta(days=2)

    next_friday = friday + timedelta(days=7)
    next_sunday = next_friday + timedelta(days=2)

    return (friday, sunday), (next_friday, next_sunday)


# === GENEROWANIE KUPONÓW ===
def generate_coupons(predictions: pd.DataFrame):
    coupons = []
    coupon_id = 1

    # grupujemy po meczu
    for (date, home, away), group in predictions.groupby(
        ["Date", "HomeTeam", "AwayTeam"]
    ):
        group = group.sort_values("Probability", ascending=False)
        group = group[group["Probability"] >= MIN_PROBABILITY]
        group = group.head(MAX_BETS_PER_MATCH)

        if group.empty:
            continue

        for _, row in group.iterrows():
            coupons.append({
                "CouponID": coupon_id,
                "Date": str(date),
                "HomeTeam": home,
                "AwayTeam": away,
                "League": row["League"],
                "Market": row["Market"],
                "Probability": round(row["Probability"], 3),
                "ModelAccuracy": round(row["ModelAccuracy"], 3),
                "GeneratedAt": datetime.utcnow().isoformat(),
                "Status": "active"
            })

            if len(coupons) % MAX_BETS_PER_COUPON == 0:
                coupon_id += 1

    return coupons


# === ARCHIWIZACJA ===
def archive_finished_coupons(coupons):
    now = datetime.utcnow()

    active = []
    for c in coupons:
        match_date = datetime.fromisoformat(c["Date"])
        if match_date < now:
            c["Status"] = "archived"
            fname = ARCHIVE_DIR / f"coupon_{c['CouponID']}.json"
            with open(fname, "w") as f:
                json.dump(c, f, indent=2)
        else:
            active.append(c)

    return active


# === MAIN ===
def run():
    print("[INFO] Loading football data...")
    df = load_football_data()

    upcoming = get_upcoming_matches(df)

    if upcoming.empty:
        print("[WARN] No upcoming matches")
        return

    print("[INFO] Predicting markets...")
    predictions = predict_markets(upcoming)

    if predictions.empty:
        print("[WARN] No predictions")
        return

    coupons = generate_coupons(predictions)
    coupons = archive_finished_coupons(coupons)

    with open(COUPONS_FILE, "w") as f:
        json.dump(coupons, f, indent=2)

    print(f"[INFO] Coupons saved: {len(coupons)}")


if __name__ == "__main__":
    run()
