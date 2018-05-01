import pandas as pd
import numpy as np
import sklearn.model_selection
import keras
import data_feeder_2 as data_feeder
import itertools
from itertools import tee, izip
import math
import matplotlib.pyplot as plt

# Focussed TDNN

target_window = 512
exog_dim = 5
epochs = 400

# def _plot(x, y, title):
#     plt.figure()
#     plt.title(title, fontsize=20)
#     plt.ylabel('Mean Square Error', fontsize=20)
#     plt.plot(x, y, 'r-')
#     plt.xlabel('Epochs', fontsize=20)
#     plt.yticks(np.arange(0, 2.2, 0.2))
#     plt.show()
#     return


def get_windowed_data_en(data, window_size):
    # local function
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)

    op = []
    for w in window(data, window_size):
        w = np.reshape(w, [-1])
        op.append(w)
    op = np.asarray(op)
    return op


def get_windowed_data_ex(data, window_size):
    # local function
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)

    exg_dim = data.shape[-1]
    op = []
    for w in window(data, window_size):
        w = np.reshape(w, [-1, exg_dim])
        op.append(w)
    op = np.asarray(op)
    return op


def FTDNN_model(window_size):
    model = keras.models.Sequential()

    # Add in layers
    num_layers = int(math.log(window_size, 2))
    print int(math.floor(num_layers))

    for l in range(num_layers):
        if l == 0:
            num_units = target_window
            layer = keras.layers.Dense(
                units=num_units,
                input_shape=[window_size],
                activation='tanh')
        else:
            num_units = int(prev_units / 4)
            layer = keras.layers.Dense(
                units=num_units,
                activation='tanh')
        model.add(layer)

        prev_units = num_units
        print num_units

        if num_units < 5:
            layer = keras.layers.Dense(
                units=1,
                activation='tanh')
            model.add(layer)
            break

    model.compile(optimizer='adam',
                  loss='mse'
                  )

    # add final layer
    # model.summary()

    return model


def FTDNN():
    global target_window
    batch_size = 128
    global epochs

    X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = data_feeder.get_data_val(True)

    train_windowed_data = get_windowed_data_ex(Y_train, target_window + 1)
    test_windowed_data = get_windowed_data_ex(Y_test, target_window + 1)
    val_windowed_data = get_windowed_data_ex(Y_val, target_window + 1)

    # Set up training input and output
    x_train = []
    y_train = []

    for i in range(train_windowed_data.shape[0]):
        x_train.append(train_windowed_data[i, 0:target_window])
        y_train.append(train_windowed_data[i, -1:])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_train = np.asarray(x_train)
    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]])
    y_train = np.asarray(y_train)
    y_train = np.reshape(y_train, [y_train.shape[0], y_train.shape[1]])

    x_val = []
    y_val = []

    for i in range(val_windowed_data.shape[0]):
        x_val.append(val_windowed_data[i, 0:target_window])
        y_val.append(val_windowed_data[i, -1:])

    x_val = np.asarray(x_val)
    x_val = np.reshape(x_val, [x_val.shape[0], x_val.shape[1]])
    y_val = np.asarray(y_val)
    y_val = np.reshape(y_val, [y_val.shape[0], y_val.shape[1]])


    x_test = []
    y_test = []

    for i in range(test_windowed_data.shape[0]):
        x_test.append(test_windowed_data[i, 0:target_window])
        y_test.append(test_windowed_data[i, -1:])

    x_test = np.asarray(x_test)
    x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1]])
    y_test = np.asarray(y_test)
    y_test = np.reshape(y_test, [y_test.shape[0], y_test.shape[1]])

    model = FTDNN_model(target_window)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size
    )
    train_loss = np.mean(history.history['loss'])

    val_mse = model.evaluate(
        x_val,
        y_val,
        batch_size=1
    )

    test_mse = model.evaluate(
        x_test,
        y_test,
        batch_size=1
    )

    return train_loss, val_mse, test_mse



def experiment():
    global target_window
    columns = ['Window Length',
               'Train Error',
               'Validation Error',
               'Test Error'
               ]
    df = pd.DataFrame(columns=columns)



    _window_length = [32, 64 ,128, 256, 512, 1024]
    for w in _window_length:
        res_dict = {}

        target_window = w
        train_loss, val_mse ,test_mse = FTDNN()

        res_dict[columns[0]] = w
        res_dict[columns[1]] = train_loss
        res_dict[columns[2]] = val_mse
        res_dict[columns[3]] = test_mse
        df = df.append(res_dict, ignore_index=True)

    df.to_csv('narx_1_op.csv')
    return


experiment()





