import pandas as pd
import numpy as np
import config
import sklearn.model_selection
from sklearn import preprocessing


def get_data(std=False):
    df = pd.read_csv('data/nasdaq100_padding.csv', header=0, index_col=None)
    nasdaq_attr = 'NDX'
    cols = list(df.columns)
    data = []
    # the target series is 'NDX'
    cols.remove(nasdaq_attr)
    scaler_array = []

    for c in cols:
        x = df[c]
        if std is True:
            scaler = preprocessing.StandardScaler().fit(x.reshape(-1, 1))
            x = scaler.transform(x.reshape(-1, 1))
            scaler_array.append(scaler)
        data.append(x)

    # X is the driving series
    X = np.asarray(data)
    print X.shape
    X = np.reshape(X, [X.shape[0], X.shape[1]])
    X = np.transpose(X, [1, 0])
    # Y is the target series

    Y = np.asarray(list(df[nasdaq_attr]))

    if std is True:
        scaler = preprocessing.StandardScaler().fit(Y.reshape(-1, 1))
        Y = scaler.transform(Y.reshape(-1, 1))
        scaler_array.append(scaler)


    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=config.test_size)

    Y_train = np.reshape(Y_train, [Y_train.shape[0], 1])
    Y_test = np.reshape(Y_test, [Y_test.shape[0], 1])
    return X_train, X_test, Y_train, Y_test , scaler_array
