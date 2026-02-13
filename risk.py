import numpy as np

def kelly(prob, odds, fraction=0.5):
    b = odds - 1
    k = (prob * b - (1 - prob)) / b
    return max(k * fraction, 0)

def monte_carlo_bankroll(selections, bankroll=1000, sims=5000):
    results = []

    for _ in range(sims):
        br = bankroll
        for _, row in selections.iterrows():
            stake = kelly(row["Probability"], row["Odds"]) * br
            outcome = np.random.rand() < row["Probability"]
            if outcome:
                br += stake * (row["Odds"] - 1)
            else:
                br -= stake
        results.append(br)

    return np.mean(results), np.percentile(results, 5)
