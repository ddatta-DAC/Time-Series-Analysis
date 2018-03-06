import pandas as pd
import numpy as np
import sklearn.model_selection

def get_data():
    df = pd.read_csv('data/nasdaq100_padding.csv', header=0, index_col=None)
    nasdaq_attr = 'NDX'
    cols = list(df.columns)
    data = []
    print cols
    cols.remove(nasdaq_attr)
    for c in cols:
        data.append(df[c])
    Y = np.asarray(data)
    Y = np.transpose(Y,[1,0])
    X = np.asarray(list(df[nasdaq_attr]))

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y , test_size=0.15 )

    print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
    return X_train, X_test, Y_train, Y_test

get_data()




