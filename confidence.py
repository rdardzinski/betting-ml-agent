import numpy as np

def calculate_confidence(row):
    ev_score = min(row["EV"] * 10, 1)
    prob_score = row["Probability"]
    stability = min(row.get("stability_index", 0.5), 1)

    score = (0.4 * prob_score) + (0.4 * ev_score) + (0.2 * stability)
    return round(score * 100, 1)
