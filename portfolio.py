import pandas as pd

def build_portfolio(selections, n=20):
    coupons = []
    selections = selections.sample(frac=1)

    for i in range(n):
        sample = selections.sample(min(4, len(selections)))
        coupons.append(sample)

    return coupons
