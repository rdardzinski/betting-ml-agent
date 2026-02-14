import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# ========================
# 0️⃣ UTWORZENIE FOLDERU
# ========================
os.makedirs("models", exist_ok=True)

# ========================
# 1️⃣ PIŁKA NOŻNA – OVER 2.5
# ========================
football_urls = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",  # Premier League
    "https://www.football-data.co.uk/mmz4281/2324/D1.csv",  # Bundesliga 2
    "https://www.football-data.co.uk/mmz4281/2324/SP1.csv", # La Liga
    "https://www.football-data.co.uk/mmz4281/2324/I1.csv",  # Serie A
    "https://www.football-data.co.uk/mmz4281/2324/F1.csv",  # Ligue 1
]

frames = []
for url in football_urls:
    try:
        df = pd.read_csv(url)
        df = df[["FTHG","FTAG"]].dropna()
        frames.append(df)
    except Exception as e:
        print(f"Nie udało się pobrać {url}: {e}")

football = pd.concat(frames, ignore_index=True)
football["Over25"] = (football["FTHG"] + football["FTAG"] > 2).astype(int)

X = football[["FTHG","FTAG"]]
y = football["Over25"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model_over25 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model_over25.fit(X_train, y_train)
acc_over25 = accuracy_score(y_test, model_over25.predict(X_test))

joblib.dump({"model": model_over25, "accuracy": acc_over25}, "models/model_over25.pkl")
print(f"Model piłki nożnej zapisany. Accuracy: {acc_over25:.3f}")

# ========================
# 2️⃣ NBA – HOME WIN
# ========================
nba_data = []
nba_url = "https://www.balldontlie.io/api/v1/games?per_page=100"

try:
    resp = requests.get(nba_url)
    if resp.status_code == 200:
        data = resp.json()["data"]
        for g in data:
            nba_data.append({
                "HomeTeam": g["home_team"]["full_name"],
                "AwayTeam": g["visitor_team"]["full_name"],
                "HomeScore": g["home_team_score"],
                "AwayScore": g["visitor_team_score"],
                "Date": g["date"][:10],
            })
    else:
        print(f"Nie udało się pobrać danych NBA: status_code={resp.status_code}")
except Exception as e:
    print(f"Błąd pobierania danych NBA: {e}")

nba = pd.DataFrame(nba_data)
if not nba.empty:
    nba["HomeWin"] = (nba["HomeScore"] > nba["AwayScore"]).astype(int)

    X = nba[["HomeScore","AwayScore"]]
    y = nba["HomeWin"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    model_nba = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model_nba.fit(X_train, y_train)
    acc_nba = accuracy_score(y_test, model_nba.predict(X_test))

    joblib.dump({"model": model_nba, "accuracy": acc_nba}, "models/model_nba.pkl")
    print(f"Model NBA zapisany. Accuracy: {acc_nba:.3f}")
else:
    print("Brak danych NBA, model nie został zapisany.")
