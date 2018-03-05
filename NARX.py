import pandas as pd
import numpy as np


def get_data():
    df = pd.read_csv('data/nasdaq100_padding.csv', header=0, index_col=None)
    print df.columns
    nasdaq_attr = 'NDX'
    Y = list(df[nasdaq_attr])
    X = range(len(nasdaq_vals))
    return X, Y


