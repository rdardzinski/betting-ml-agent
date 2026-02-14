import json
import pandas as pd
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
from predictor import predict

# =========================
# GLOBALNA STRUKTURA
# =========================
BASE_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "League", "Sport",
    "Over25_Prob", "HomeWin_Prob", "ValueScore"
]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Gwarantuje stałą strukturę danych"""
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


# =========================
# GENEROWANIE KUPONÓW
# =========================
def generate_coupons(df, n=5, size=5):
    if df.empty:
        return []

    df = df.sort_values("ValueScore", ascending=False).reset_index(drop=True)
    coupons = []

    for i in range(n):
        block = df.iloc[i*size:(i+1)*size]
        if not block.empty:
            coupons.append(block.index.tolist())

    return coupons


# =========================
# AGENT
# =========================
def run():
    frames = []

    # ---------- FOOTBALL ----------
    try:
        football = get_next_matches()
        print(f"Football matches: {len(football)}")

        if not football.empty:
            football = build_features(football)
            football_pred, = predict(football)

            football_pred["Sport"] = "Football"
            football_pred["ValueScore"] = football_pred.get("Over25_Prob", 0)

            frames.append(normalize(football_pred))

    except Exception as e:
        print("[ERROR] Football:", e)

    # ---------- BASKETBALL ----------
    try:
        basket = get_basketball_games()
        print(f"Basketball matches: {len(basket)}")

        if not basket.empty:
            basket["Sport"] = "Basketball"
            basket["HomeWin_Prob"] = 0.55
            basket["ValueScore"] = basket["HomeWin_Prob"]

            frames.append(normalize(basket))

    except Exception as e:
        print("[ERROR] Basketball:", e)

    # ---------- FINAL ----------
    if not frames:
        print("[FATAL] No data")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize(combined)

    combined.to_csv("predictions.csv", index=False)
    print("Predictions saved:", len(combined))

    coupons = generate_coupons(combined)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print("Coupons saved:", len(coupons))
    print("Agent finished successfully")


if __name__ == "__main__":
    run()
