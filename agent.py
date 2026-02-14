import pandas as pd
import json
import os

from data_loader import get_next_matches
from feature_engineering import build_features
from predictor import predict
from evaluation import update_history

def run():
    print("[INFO] Loading football matches...")
    football = get_next_matches()
    print(f"[INFO] Football matches loaded: {len(football)}")

    if football.empty:
        print("[WARN] No football data – exiting")
        return

    football = build_features(football)

    # 🔧 KLUCZOWA POPRAWKA TUTAJ
    preds_df, meta = predict(football)

    # łączymy predykcje z danymi
    football = football.reset_index(drop=True)
    preds_df = preds_df.reset_index(drop=True)
    football = pd.concat([football, preds_df], axis=1)

    # zapis predykcji
    football.to_csv("predictions.csv", index=False)
    print(f"[INFO] Predictions saved: {len(football)} rows")

    # historia skuteczności
    update_history(football)

    # -----------------------------
    # 🏀 BASKETBALL – WYŁĄCZONY
    # -----------------------------
    """
    from data_loader_basketball import get_basketball_games
    basketball = get_basketball_games()
    """

    # kupony – tylko football
    value_bets = football[football["Over25"] > 0.55]
    coupons = []

    coupon_size = 4
    for i in range(0, len(value_bets), coupon_size):
        coupons.append(value_bets.index[i:i+coupon_size].tolist())

    with open("coupons.json", "w") as f:
        json.dump(coupons[:5], f)

    print(f"[INFO] Coupons saved: {len(coupons[:5])}")
    print("[INFO] Agent finished successfully")

if __name__ == "__main__":
    run()
