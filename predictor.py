# predictor.py
def predict(match):
    """
    match: pd.Series z kolumnami HomeTeam, AwayTeam, League, Date
    zwraca listę słowników:
        Market, Probability, ModelAccuracy, ValueFlag
    """
    markets = ['Over25', 'BTTS', '1HGoals', '2HGoals', 'Cards', 'Corners']
    preds = []
    for m in markets:
        pred = {
            'Market': m,
            'Probability': 0.5,   # placeholder
            'ModelAccuracy': 0.8, # placeholder
            'ValueFlag': False     # placeholder
        }
        preds.append(pred)
    return preds
