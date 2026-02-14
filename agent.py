import json
import pandas as pd
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
from predictor import predict

# =========================
# UTILS
# =========================

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "League", "Sport"]

def ensure_columns(df):
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = None
    return df


# =========================
# GENEROWANIE KUPONÓW
# =========================

def generate_coupons(df, n_coupons=5, picks=5):
    if df.empty:
        return []

    df = df.sort_values("ValueScore", ascending=False).reset_index(drop=True)

    coupons = []
    for i in range(n_coupons):
        chunk = df.iloc[i*picks:(i+1)*picks]
        if not chunk.empty:
            coupons.append(chunk.index.tolist())

    return coupons


# =========================
# GŁÓWNA LOGIKA AGENTA
# =========================

def run():
    all_frames = []

    # -------------------------
    # FOOTBALL
    # -------------------------
    try:
        football = get_next_matches()
        print(f"Football matches loaded: {len(football)}")

        if not football.empty:
            football = build_features(football)
            football_pred, = predict(football)

            football_pred["Sport"] = "Football"
            football_pred["ValueScore"] = football_pred.get("Over25_Prob", 0)

            football_pred = ensure_columns(football_pred)
            all_frames.append(football_pred)

    except Exception as e:
        print("[ERROR] Football pipeline failed:", e)

    # -------------------------
    # BASKETBALL
    # -------------------------
    try:
        basketball = get_basketball_games()
        print(f"Basketball matches loaded: {len(basketball)}")

        if not basketball.empty:
            basketball["HomeWin_Prob"] = 0.55  # free-first baseline
            basketball["HomeWin_ModelAccuracy"] = 0.55
            basketball["ValueScore"] = basketball["HomeWin_Prob"]
            basketball["Sport"] = "Basketball"

            basketball = ensure_columns(basketball)
            all_frames.append(basketball)

    except Exception as e:
        print("[ERROR] Basketball pipeline failed:", e)

    # -------------------------
    # ŁĄCZENIE
    # -------------------------
    if not all_frames:
        print("[FATAL] No data generated")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = ensure_columns(combined)

    combined.to_csv("predictions.csv", index=False)
    print(f"Predictions saved: {len(combined)} rows")

    # -------------------------
    # KUPONY
    # -------------------------
    coupons = generate_coupons(combined)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"Coupons saved: {len(coupons)}")
    print("Agent finished successfully")


if __name__ == "__main__":
    run()
