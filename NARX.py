# Time Delay Neural Networks

import pandas as pd
import numpy as np
import sklearn.model_selection
import keras
import data_feeder
import itertools
from itertools import tee, izip
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import Flatten

# Focussed TDNN

target_window = 4096


def _plot(x, y, title):
    plt.figure()
    plt.title(title, fontsize=20)
    plt.ylabel('Mean Square Error', fontsize=18)
    plt.plot(x, y, 'r-')
    plt.xlabel('Epochs', fontsize=18)
    plt.yticks(np.arange(0, 2.2, 0.2))
    plt.show()
    return


def get_windowed_data_ex(data, window_size):
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
    X_train, X_test, Y_train, Y_test, scaler_array = data_feeder.get_data(True)

    # print X_train.shape
    # print Y_train.shape
    train_windowed_data = get_windowed_data_ex(Y_train, target_window + 1)
    test_windowed_data = get_windowed_data_ex(Y_test, target_window + 1)
    print train_windowed_data.shape
    # Set up training input and output
    x_train = []
    y_train = []

    for i in range(train_windowed_data.shape[0]):
        x_train.append(train_windowed_data[i, 0:target_window])
        y_train.append(train_windowed_data[i, -1:])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    print x_train.shape
    print y_train.shape

    x_test = []
    y_test = []

    for i in range(test_windowed_data.shape[0]):
        x_test.append(test_windowed_data[i, 0:target_window])
        y_test.append(test_windowed_data[i, -1:])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print x_test.shape
    print y_test.shape

    model = FTDNN_model(target_window)
    history = model.fit(
        x_train,
        y_train,
        epochs=400,
        batch_size=128
    )
    train_loss = history.history['loss']
    print train_loss
    # Plot Training Loss
    # _plot(range(len(train_loss)), train_loss,
    #        ' Training Loss in Focused TDNN with target series input only. Batch size of 128')
    score = model.evaluate(x_test, y_test, batch_size=128)
    print score


# FTDNN()

# ------------------------------ #

def conv_aux(window_size,i):
    exog_inp = Input(shape=[window_size,1], dtype='float32', name= 'exg_inp_'+str(i))
    conv1_op = Conv1D(filters=1, kernel_size=32, activation='relu')(exog_inp)
    pool1_op = MaxPooling1D(pool_size=16)(conv1_op)
    # conv2_op = Conv1D(filters=1, kernel_size=16, activation='relu')(pool1_op)
    # pool2_op = MaxPooling1D(pool_size=2)(conv2_op)
    pool2_op = keras.layers.Flatten()(pool1_op)
    output = Dense(1, activation='tanh')(pool2_op)
    return exog_inp, output


def FTDNN_conv_model(window_size, exog_dim):

    exg_conv_op = []
    input = []
    for i in range(exog_dim):
        exg_inp, ex_op = conv_aux(window_size, i)
        input.append(exg_inp)
        exg_conv_op.append(ex_op)

    exg_1 = keras.layers.concatenate(exg_conv_op)
    print exg_1
    end_inp = Input(shape=[window_size])
    input.append(end_inp)
    layer_1 = Dense(16, activation='tanh')(end_inp)
    # layer_2 = Dense(16, activation='tanh')(layer_1)

    op_1 = keras.layers.concatenate(inputs = [exg_1, layer_1])
    layer_3 = Dense(8, activation='tanh')(op_1)
    output = Dense(1, activation='tanh')(layer_3)
    return input, output



def _get_data(window_size):


    exog_train, exog_test, end_train, end_test, scaler_array = data_feeder.get_data(True)

    # print X_train.shape
    # print Y_train.shape
    train_windowed_data_exg = get_windowed_data_ex(exog_train, target_window + 1)
    test_windowed_data_exg = get_windowed_data_ex(exog_test, target_window + 1)

    train_windowed_data_end = get_windowed_data_ex(end_train, target_window + 1)
    test_windowed_data_end = get_windowed_data_ex(end_test, target_window + 1)

    print test_windowed_data_end.shape
    print test_windowed_data_exg.shape

    return

    # Set up training input and output
    x_train = []
    y_train = []

    for i in range(train_windowed_data.shape[0]):
        x_train.append(train_windowed_data[i, 0:target_window])
        y_train.append(train_windowed_data[i, -1:])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    print x_train.shape
    print y_train.shape

    x_test = []
    y_test = []

    for i in range(test_windowed_data.shape[0]):
        x_test.append(test_windowed_data[i, 0:target_window])
        y_test.append(test_windowed_data[i, -1:])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print x_test.shape
    print y_test.shape


def FTDNN_conv():
    global target_window
    window_size = target_window


    model_in , model_op = FTDNN_conv_model(window_size, 81)
    x_inputs = []

    for i in range(81):
        x_inputs.append(np.random.rand(1000, window_size,1))
    x_inputs.append(np.random.rand(1000, window_size))
    y_inp = np.random.rand(1000,1)
    print len(x_inputs)
    model = Model(inputs=model_in, outputs=model_op)
    model.compile(
        optimizer='adam',
        loss = 'mse'
    )
    # model.summary()
    model.fit(x=x_inputs,y=y_inp)


# FTDNN_conv()
_get_data(128)