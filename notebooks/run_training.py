import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Upewnij się, że folder models istnieje
os.makedirs("models", exist_ok=True)

# =======================
# PIŁKA NOŻNA – OVER 2.5
# =======================
urls = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
]

frames = []
for url in urls:
    df = pd.read_csv(url)
    df = df[["FTHG","FTAG"]].dropna()
    frames.append(df)

football = pd.concat(frames, ignore_index=True)
football["Over25"] = (football["FTHG"] + football["FTAG"] > 2).astype(int)

X = football[["FTHG","FTAG"]]
y = football["Over25"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model_over25 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model_over25.fit(X_train,y_train)
acc = accuracy_score(y_test, model_over25.predict(X_test))

# Zapis modelu
joblib.dump({"model":model_over25,"accuracy":acc},"models/model_over25.pkl")

# =======================
# NBA – HOME WIN
# =======================
nba_url = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
nba = pd.read_csv(nba_url)
nba = nba[["home_points","visitor_points"]].dropna()
nba["HomeWin"] = (nba["home_points"] > nba["visitor_points"]).astype(int)

X = nba[["home_points","visitor_points"]]
y = nba["HomeWin"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model_nba = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model_nba.fit(X_train,y_train)
acc_nba = accuracy_score(y_test, model_nba.predict(X_test))

joblib.dump({"model":model_nba,"accuracy":acc_nba},"models/model_nba.pkl")
