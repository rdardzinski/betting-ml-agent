import pandas as pd

def build_features(df):
    df = df.copy()

    # --- Rolling goals ---
    for team in ["Home","Away"]:
        df[f"{team}RollingGoals"] = df.groupby(f"{team}Team")["FTHG" if team=="Home" else "FTAG"]\
                                      .rolling(5,min_periods=1).mean().reset_index(0,drop=True)

    # --- BTTS ---
    df["BTTS"] = ((df["FTHG"]>0) & (df["FTAG"]>0)).astype(int)

    # --- Halftime/Second half goals ---
    if "HTHG" in df.columns and "HTAG" in df.columns:
        df["1HGoals"] = df["HTHG"] + df["HTAG"]
        df["2HGoals"] = df["FTHG"] + df["FTAG"] - df["1HGoals"]
    else:
        df["1HGoals"] = df["FTHG"]//2
        df["2HGoals"] = df["FTAG"]//2

    # --- Cards / Corners (fallback if not present) ---
    df["Cards"] = df.get("HY",0) + df.get("AY",0)
    df["Corners"] = df.get("HC",0) + df.get("AC",0)

    # --- Feature selection for model ---
    feature_cols = ["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals",
                    "1HGoals","2HGoals","BTTS","Cards","Corners"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]
