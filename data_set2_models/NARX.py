# Time Delay Neural Networks

import os
os.chdir('./..')

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
from keras.models import load_model
# Focussed TDNN

target_window = 512


def _plot(x, y, title):
    plt.figure()
    plt.title(title, fontsize=20)
    plt.ylabel('Mean Square Error', fontsize=20)
    plt.plot(x, y, 'r-')
    plt.xlabel('Epochs', fontsize=20)
    plt.yticks(np.arange(0, 2.2, 0.2))
    plt.show()
    return

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
    batch_size = 256

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

    x_train = np.asarray(x_train)
    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1]])
    y_train = np.asarray(y_train)
    y_train = np.reshape(y_train, [y_train.shape[0], y_train.shape[1]])


    print x_train.shape
    print y_train.shape

    x_test = []
    y_test = []

    for i in range(test_windowed_data.shape[0]):
        x_test.append(test_windowed_data[i, 0:target_window])
        y_test.append(test_windowed_data[i, -1:])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)



    x_test = np.asarray(x_test)
    x_test = np.reshape(x_test,[x_test.shape[0],x_test.shape[1]])
    y_test = np.asarray(y_test)
    y_test = np.reshape(y_test, [y_test.shape[0], y_test.shape[1]])

    print x_train.shape
    print y_test.shape

    model = FTDNN_model(target_window)
    history = model.fit(
        x_test,
        y_train,
        epochs=2,
        batch_size=batch_size
    )
    train_loss = history.history['loss']
    print train_loss
    # Plot Training Loss
    # _plot(range(len(train_loss)), train_loss,
    #        ' Training Loss in Focused TDNN with target series input only. Batch size of 128')
    score = model.evaluate(x_test, y_test, batch_size=128)
    print score


FTDNN()


exit(1)


# ------------------------------ #

def conv_aux(window_size, i):
    exog_inp = Input(shape=[window_size, 1], dtype='float32', name='exg_inp_' + str(i))
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

    op_1 = keras.layers.concatenate(inputs=[exg_1, layer_1])
    layer_3 = Dense(8, activation='tanh')(op_1)
    output = Dense(1, activation='tanh')(layer_3)
    return input, output


def ctdnn_get_data_aux(exog, end, window_size):

    t_windowed_data_exg = get_windowed_data_ex(exog, window_size + 1)
    t_windowed_data_end = get_windowed_data_en(end, window_size + 1)

    exog_t = []
    end_t = []
    y_t = []

    for i in range(t_windowed_data_end.shape[0]):
        exog_t.append(t_windowed_data_exg[i, 0:window_size])
        end_t.append(t_windowed_data_end[i, 0:window_size])
        y_t.append(t_windowed_data_end[i, -1:])

    exog_t = np.asarray(exog_t)
    end_t = np.asarray(end_t)
    y_t = np.asarray(y_t)

    exg_dim = exog_t.shape[-1]
    exog_t = np.transpose(exog_t, [0, 2, 1])

    exog_t_inputs = []
    for d in range(exg_dim):
        _tmp = exog_t[:, d, :]
        _tmp = np.reshape(_tmp, [-1, window_size, 1])
        exog_t_inputs.append(_tmp)

    return exog_t_inputs, end_t, y_t


def ctdnn_get_data(window_size, type='trian'):
    exog_train, exog_test, end_train, end_test, scaler_array = data_feeder.get_data(True)

    if type == 'train':
        train_x_exog, train_x_end, train_y = ctdnn_get_data_aux(exog_train, end_train, window_size)
        return train_x_exog, train_x_end, train_y
    else:
        test_x_exog, test_x_end, test_y = ctdnn_get_data_aux(exog_test, end_test, window_size)
        return test_x_exog, test_x_end, test_y


def FTDNN_conv():
    global target_window
    window_size = target_window

    model_in, model_op = FTDNN_conv_model(window_size, 81)
    x_inputs = []

    for i in range(81):
        x_inputs.append(np.random.rand(1000, window_size, 1))
    x_inputs.append(np.random.rand(1000, window_size))
    y_inp = np.random.rand(1000, 1)
    print len(x_inputs)



    model = Model(inputs=model_in, outputs=model_op)
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    train_x_exog, train_x_end, train_y = ctdnn_get_data( window_size, 'train')
    x_inputs_train = train_x_exog
    x_inputs_train.append(train_x_end)

    # model.summary()
    h = model.fit(x= x_inputs_train, y=train_y, epochs = 150 , batch_size= 512)
    train_loss = h.history['loss']
    _plot(range(len(train_loss)),
          train_loss,
        ' Training Loss in Focused TDNN with a 1D Convolution. Batch size of 512')

    save_file = 'convTDNN.h5'
    model.save(save_file)
    test_x_exog, test_x_end, test_y = ctdnn_get_data(window_size, 'test')
    x_inputs_test = test_x_exog
    x_inputs_test.append(test_x_end)

    score = model.evaluate(x_inputs_test, test_y, batch_size=512)
    print score

# FTDNN_conv()
ctdnn_get_data(128)
FTDNN_conv()