import requests
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

os.makedirs("models", exist_ok=True)

# =======================
# PIŁKA NOŻNA – OVER 2.5
# =======================
# (tu zostaje Twój kod dla football – bez zmian)
# ...

# =======================
# NBA – HOME WIN (BALLEDONTLIE)
# =======================
nba_data = []
url = "https://www.balldontlie.io/api/v1/games?per_page=100"
resp = requests.get(url)
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

nba = pd.DataFrame(nba_data)
nba["HomeWin"] = (nba["HomeScore"] > nba["AwayScore"]).astype(int)

X = nba[["HomeScore","AwayScore"]]
y = nba["HomeWin"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
model_nba = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model_nba.fit(X_train, y_train)
acc_nba = accuracy_score(y_test, model_nba.predict(X_test))

joblib.dump({"model":model_nba,"accuracy":acc_nba},"models/model_nba.pkl")
