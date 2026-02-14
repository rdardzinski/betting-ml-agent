import json
import pandas as pd

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict

# from data_loader_basketball import get_basketball_games  # ⛔ POMINIĘTE
# from predictor_basketball import predict_basketball     # ⛔ POMINIĘTE


# =========================
# KONFIGURACJA
# =========================
FOOTBALL_MARKETS = [
    "Over25",
    "BTTS",
    "Over15",
    "Under35",
    "HomeTeamScore",
    "AwayTeamScore",
    "TotalGoals"
]

BASE_COLS = [
    "Date", "League", "HomeTeam", "AwayTeam", "Sport", "ValueScore"
]


# =========================
# UTILS
# =========================
def ensure_columns(df):
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def generate_coupons(df, coupons=5, picks=5):
    if df.empty:
        return []

    df = df.sort_values("ValueScore", ascending=False).reset_index(drop=True)
    result = []

    for i in range(coupons):
        block = df.iloc[i*picks:(i+1)*picks]
        if not block.empty:
            result.append(block.index.tolist())

    return result


# =========================
# AGENT
# =========================
def run():
    print("[INFO] Loading football matches...")
    matches = get_next_matches()
    print(f"[INFO] Football matches loaded: {len(matches)}")

    if matches.empty:
        print("[FATAL] No football data")
        return

    # ZAPAMIĘTUJ NAZWY DRUŻYN
    meta = matches[["Date", "League", "HomeTeam", "AwayTeam"]].copy()

    # FEATURE ENGINEERING
    features = build_features(matches)

    # PREDYKCJE (tylko liczby)
    preds, = predict(features)

    # 🔴 KLUCZOWY MOMENT – DOKLEJANIE NAZW
    football = preds.copy()
    football[["Date", "League", "HomeTeam", "AwayTeam"]] = meta

    football["Sport"] = "Football"

    # =========================
    # RYNKI PIŁKARSKIE
    # =========================
    football["Over25"] = football.get("Over25_Prob", 0)
    football["BTTS"] = football.get("BTTS_Prob", 0)

    football["Over15"] = football["Over25"].clip(lower=0.60)
    football["Under35"] = (1 - football["Over25"]).clip(lower=0.55)

    football["HomeTeamScore"] = football["Over25"] * 0.8
    football["AwayTeamScore"] = football["Over25"] * 0.7
    football["TotalGoals"] = football["Over25"]

    # VALUE SCORE – JEDNA METRYKA
    football["ValueScore"] = football[
        ["Over25", "BTTS", "Over15"]
    ].max(axis=1)

    football = ensure_columns(football)

    # =========================
    # ZAPIS
    # =========================
    football.to_csv("predictions.csv", index=False)
    print(f"[INFO] Predictions saved: {len(football)} rows")

    coupons = generate_coupons(football)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"[INFO] Coupons saved: {len(coupons)}")
    print("[SUCCESS] Agent finished")


if __name__ == "__main__":
    run()
