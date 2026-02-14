import pandas as pd
import json
from predictor import predict
from feature_engineering import build_features
from data_loader import get_next_matches
# from data_loader_basketball import get_basketball_games  # aktualnie pomijamy koszykówkę

def generate_coupons(df, n_coupons=5, picks=5):
    """
    Generuje kupony po `picks` meczów.
    Każdy zakład = jeden mecz, niezależnie od liczby typów.
    """
    df_sorted = df.sort_values("ValueScore", ascending=False).reset_index()
    coupons = []

    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        if start >= len(df_sorted):
            break
        coupons.append(df_sorted.loc[start:end-1, "index"].tolist())
    return coupons

def run():
    # --- PIŁKA NOŻNA ---
    football = get_next_matches()
    if football.empty:
        print("[WARN] Brak meczów piłki nożnej!")
        return

    football = build_features(football)
    football_pred = predict(football)

    # Obliczamy ValueScore np. średnia probabilistyczna dla wszystkich rynków
    prob_cols = [col for col in football_pred.columns if "_Prob" in col]
    football_pred["ValueScore"] = football_pred[prob_cols].mean(axis=1)

    # --- opcjonalnie: KOSZYKÓWKA ---
    # basketball = get_basketball_games()
    # basketball_pred = ...  # przetwarzanie koszykówki

    # --- ZAPIS ---
    football_pred.to_csv("predictions.csv", index=False)

    coupons = generate_coupons(football_pred)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)

    print(f"[INFO] Football matches: {len(football_pred)}")
    print(f"[INFO] Coupons saved: {len(coupons)}")
    print("Agent finished successfully")

if __name__ == "__main__":
    run()
