def rank_markets(selections):
    ranking = selections.groupby("Market")["EV"].mean().sort_values(ascending=False)
    return ranking
