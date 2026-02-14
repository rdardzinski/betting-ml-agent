def predict(df):
    predictions = df.copy()  # <-- zachowujemy wszystkie kolumny
    markets = ["Over25","BTTS","1HGoals","2HGoals","Cards","Corners"]

    for market in markets:
        try:
            model, accuracy = load_model(market)
            features = df[["FTHG","FTAG","HomeRollingGoals","AwayRollingGoals"]]
            probs_raw = model.predict_proba(features)
            probs = probs_raw[:,1] if probs_raw.shape[1] > 1 else np.full(len(features), probs_raw[0,0])
            predictions[f"{market}_Prob"] = probs
            predictions[f"{market}_Confidence"] = (probs*100).round(1)
            predictions[f"{market}_ValueFlag"] = probs > 0.55
            predictions[f"{market}_ModelAccuracy"] = accuracy
        except Exception as e:
            print(f"[WARN] Prediction for {market} failed: {e}")

    return predictions, None
