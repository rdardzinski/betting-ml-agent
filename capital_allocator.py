def allocate_capital(selections, bankroll=1000):
    league_weights = (
        selections.groupby("League")["EV"]
        .mean()
        .to_dict()
    )

    total_weight = sum(league_weights.values())

    allocations = {}

    for league, weight in league_weights.items():
        share = weight / total_weight if total_weight > 0 else 0
        allocations[league] = round(bankroll * share, 2)

    return allocations
