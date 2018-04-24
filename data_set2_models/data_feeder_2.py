import pandas as pd
import numpy as np
import config
import sklearn.model_selection
from sklearn import preprocessing
import os

print os.getcwd()
try:
    os.chdir('data_set2_models')
except :
    pass

def get_data(std=False):

    df = pd.read_csv('dataset2/weather.csv', header=0, index_col=0)
    target_attr = 'temp'
    cols = list(df.columns)
    exog_attr = list( cols )
    exog_attr.remove(target_attr)

    data = []
    scaler_array = []

    for c in exog_attr:
        x = df[c]
        if std is True:
            scaler = preprocessing.StandardScaler().fit(x.values.reshape(-1, 1))
            x = scaler.transform(x.values.reshape(-1, 1))
            scaler_array.append(scaler)
        data.append(x)
    # X is the exogenous series

    X = np.asarray(data)
    X = np.transpose(X,[1,0,2])
    X = np.reshape(X, [X.shape[0], X.shape[1]])
    Y = np.asarray(list(df[target_attr]))
    Y = Y.reshape(-1, 1)


    if std is True:
        scaler = preprocessing.StandardScaler().fit(Y.reshape(-1, 1))
        Y = scaler.transform(Y.reshape(-1, 1))
        scaler_array.append(scaler)


    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=config.test_size)
    Y_train = np.reshape(Y_train, [Y_train.shape[0], 1])
    Y_test = np.reshape(Y_test, [Y_test.shape[0], 1])


    return X_train, X_test, Y_train, Y_test, scaler_array


get_data(True)