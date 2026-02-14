import json
import pandas as pd
from data_loader import get_next_matches
from data_loader_basketball import get_basketball_games
from feature_engineering import build_features
from predictor import predict

def generate_coupons(df, n_coupons=5, picks=5):
    """
    Generuje listę kuponów po n_coupons z picks zakładami
    """
    df = df.sort_values("ValueScore", ascending=False)
    coupons = []
    indices = list(df.index)
    for i in range(n_coupons):
        start = i * picks
        end = start + picks
        if start >= len(indices):
            break  # brak wierszy
        coupons.append(indices[start:end])
    return coupons

def run():
    # ----------------------------
    # 1️⃣ PIŁKA NOŻNA
    # ----------------------------
    football = get_next_matches()
    print(f"Football matches: {len(football)}")  # debug
    if not football.empty:
        football = build_features(football)
        football_pred = predict(football)

        # Uzupełnij brakujące kolumny
        for col in ["HomeTeam", "AwayTeam", "League", "Date"]:
            football_pred[col] = football.get(col, "Unknown")

        football_pred["Sport"] = "Football"
        football_pred["Over25_Prob"] = football_pred.get("Over25_Prob", 0.5)
        football_pred["ValueScore"] = football_pred["Over25_Prob"]
    else:
        football_pred = pd.DataFrame(columns=[
            "Date","HomeTeam","AwayTeam","League","Sport","Over25_Prob","ValueScore"
        ])

    # ----------------------------
    # 2️⃣ KOSZYKÓWKA (NBA)
    # ----------------------------
    basketball = get_basketball_games(pages=3)  # pobiera 3 strony po 100 meczów
    print(f"Basketball matches: {len(basketball)}")  # debug

    if not basketball.empty:
        basketball_pred = predict(basketball)
        basketball_pred["Sport"] = "Basketball"
        basketball_pred["HomeWin_Prob"] = basketball_pred.get("HomeWin_Prob", 0.5)
        basketball_pred["ValueScore"] = basketball_pred["HomeWin_Prob"]

        # Zmiana nazw kolumn, by spójnie łączyć z football
        basketball_pred = basketball_pred.rename(columns={"HomeScore":"FTHG","AwayScore":"FTAG"})
    else:
        basketball_pred = pd.DataFrame(columns=football_pred.columns)

    # ----------------------------
    # 3️⃣ ŁĄCZENIE DANYCH
    # ----------------------------
    combined = pd.concat([football_pred, basketball_pred], ignore_index=True)

    # Uzupełnienie brakujących kolumn i wartości
    combined["ValueScore"] = combined.get("ValueScore", 0.5)
    combined["Over25_Prob"] = combined.get("Over25_Prob", 0.5)
    combined["HomeWin_Prob"] = combined.get("HomeWin_Prob", 0.5)

    # ----------------------------
    # 4️⃣ ZAPIS PREDYKCJI
    # ----------------------------
    combined.to_csv("predictions.csv", index=False)
    print(f"Predictions saved: {len(combined)} rows")

    # ----------------------------
    # 5️⃣ GENEROWANIE KUPONÓW
    # ----------------------------
    coupons = generate_coupons(combined, n_coupons=5, picks=5)
    with open("coupons.json", "w") as f:
        json.dump(coupons, f)
    print(f"Coupons saved: {len(coupons)}")

    print("Agent finished successfully")

if __name__ == "__main__":
    run()
