import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import asyncio
import aiohttp
import time
import os

# =========================
# CONSTANTS
# =========================
CUTOFF_DATE = datetime.today() - timedelta(days=180)  # ostatnie 6 miesięcy
FALLBACK_CSV = "data/basketball_fallback.csv"
SAVE_INCREMENTAL = "data/basketball_incremental.csv"

# =========================
# HELPER FUNCTIONS
# =========================
def _normalize_df(df, src):
    """Normalizuje do wspólnego formatu agent"""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["League"] = df.get("League", src)
    return df[
        ["Date","HomeTeam","AwayTeam","League","HomeScore","AwayScore"]
    ].dropna(subset=["HomeTeam","AwayTeam"])

# =========================
# 1️⃣ Sportsflakes API (top EU leagues)
# =========================
def _load_sportsflakes():
    out = []
    endpoints = [
        ("Euroleague","https://sportsflakes.com/api/basketball/euroleague/matches"),
        ("Spain_ACB","https://sportsflakes.com/api/basketball/spain-acb/matches"),
        ("Germany_BBL","https://sportsflakes.com/api/basketball/germany-bbl/matches"),
        ("Italy_LegaA","https://sportsflakes.com/api/basketball/italy-legaa/matches"),
        ("France_LNB","https://sportsflakes.com/api/basketball/france-lnb/matches")
    ]
    for league,url in endpoints:
        try:
            r = requests.get(url, timeout=10)
            time.sleep(0.5)  # delay dla regulaminu
            if r.status_code == 200:
                js = r.json()
                for g in js.get("matches", []):
                    date = pd.to_datetime(g.get("date"), errors="coerce")
                    if date and date >= CUTOFF_DATE:
                        out.append({
                            "Date": date,
                            "HomeTeam": g.get("home_team",""),
                            "AwayTeam": g.get("away_team",""),
                            "HomeScore": g.get("home_score",0),
                            "AwayScore": g.get("away_score",0),
                            "League": league
                        })
        except Exception as e:
            print(f"[WARN][Sportsflakes] {league}: {e}")
    return pd.DataFrame(out)

# =========================
# 2️⃣ NBA GitHub CSV
# =========================
def _load_nba_github():
    try:
        url = "https://raw.githubusercontent.com/bttmly/nba/master/data/games.csv"
        df = pd.read_csv(url)
        df = df[["date","home_team","visitor_team","home_points","visitor_points"]]
        df.columns = ["Date","HomeTeam","AwayTeam","HomeScore","AwayScore"]
        return _normalize_df(df,"NBA")
    except Exception as e:
        print(f"[WARN][NBA GitHub CSV] {e}")
        return pd.DataFrame()

# =========================
# 3️⃣ Livesport scraper (async)
# =========================
async def _fetch_livesport_page(session, url):
    try:
        async with session.get(url) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html,"html.parser")
            rows = soup.select(".sportName .event__match")
            out = []
            for row in rows:
                try:
                    date_str = row.select_one(".event__time").text.strip()
                    date = datetime.strptime(date_str,"%H:%M")
                    home = row.select_one(".event__participant--home").text.strip()
                    away = row.select_one(".event__participant--away").text.strip()
                    out.append({
                        "Date": date,
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "HomeScore": None,
                        "AwayScore": None,
                        "League": "Livesport"
                    })
                except:
                    continue
            return out
    except Exception as e:
        print(f"[WARN][Livesport async] {e}")
        return []

async def _load_livesport_async():
    urls = [f"https://www.livesport.com/pl/koszykowka/?page={i}" for i in range(1,4)]
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_livesport_page(session,u) for u in urls]
        results = await asyncio.gather(*tasks)
    all_matches = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_matches)
    return df

def _load_livesport():
    """Wrapper sync dla asynchronicznego scraper"""
    start = time.time()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(_load_livesport_async())
        print(f"[INFO] Livesport fetched {len(df)} matches in {time.time()-start:.1f}s")
        return _normalize_df(df,"Livesport") if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"[WARN][Livesport sync] {e}")
        return pd.DataFrame()

# =========================
# 4️⃣ Fallback CSV
# =========================
def _load_fallback_csv():
    if not os.path.exists(FALLBACK_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(FALLBACK_CSV)
        return _normalize_df(df, df.get("League","Fallback"))
    except Exception as e:
        print(f"[WARN][Fallback CSV] {e}")
        return pd.DataFrame()

# =========================
# PUBLIC FUNCTION
# =========================
def get_basketball_games():
    """
    Pobiera dane koszykarskie z kilku źródeł:
    1) Sportsflakes API (EU top 5 lig + Euroliga)
    2) NBA public CSV
    3) Livesport HTML (async)
    4) Fallback CSV lokalny
    
    Zwraca DataFrame z kolumnami:
    Date, HomeTeam, AwayTeam, League, HomeScore, AwayScore
    """
    sources = [
        ("Sportsflakes", _load_sportsflakes),
        ("NBA GitHub", _load_nba_github),
        ("Livesport", _load_livesport),
        ("Fallback CSV", _load_fallback_csv)
    ]
    final_df = pd.DataFrame()
    for name, loader in sources:
        df = loader()
        if not df.empty:
            print(f"[INFO] {name} loaded {len(df)} matches")
            final_df = pd.concat([final_df, df], ignore_index=True)
            # zapis przyrostowy
            df.to_csv(SAVE_INCREMENTAL, mode='a', index=False, header=not os.path.exists(SAVE_INCREMENTAL))
    if final_df.empty:
        print("[WARN] No basketball data found from any source")
    return final_df
