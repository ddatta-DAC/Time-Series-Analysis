import pandas as pd
import numpy as np
import sklearn.model_selection
import keras
import data_feeder_2 as data_feeder
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

window_size = 64
exog_dim = 5
batch_size = 128
epochs = 300

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


def conv_aux(window_size, i):
    exog_inp = Input(
        shape=[window_size, 1],
        dtype='float32',
        name='exg_inp_' + str(i)
    )
    conv1_op = Conv1D(
        filters=1,
        kernel_size=32,
        activation='relu')(exog_inp)
    pool1_op = MaxPooling1D(pool_size=16)(conv1_op)

    pool2_op = keras.layers.Flatten()(pool1_op)
    output = Dense(1, activation='tanh')(pool2_op)
    return exog_inp, output


def FTDNN_conv_model(window_size, exog_dim):
    global dense_layer_units

    dense_layer_units = [16, 8]
    exg_conv_op = []
    input = []
    for i in range(exog_dim):
        exg_inp, ex_op = conv_aux(window_size, i)
        input.append(exg_inp)
        exg_conv_op.append(ex_op)

    exg_1 = keras.layers.concatenate(exg_conv_op)
    end_inp = Input(shape=[window_size])
    input.append(end_inp)

    layer_1 = Dense(
        dense_layer_units[0],
        activation='tanh'
    )(end_inp)

    op_1 = keras.layers.concatenate(inputs=[exg_1, layer_1])
    layer_2 = Dense(
        dense_layer_units[1],
        activation='tanh'
    )(op_1)

    output = Dense(
        1,
        activation='tanh'
    )(layer_2)
    return input, output


def ctdnn_get_data(window_size, type='trian'):
    exog_train, exog_test, end_train, end_test, scaler_array = data_feeder.get_data(True)

    if type == 'train':
        train_x_exog, train_x_end, train_y = ctdnn_get_data_aux(exog_train, end_train, window_size)
        return train_x_exog, train_x_end, train_y
    else:
        test_x_exog, test_x_end, test_y = ctdnn_get_data_aux(exog_test, end_test, window_size)
        return test_x_exog, test_x_end, test_y


def FTDNN_conv():

    global window_size
    global exog_dim
    global batch_size
    global epochs

    model_in, model_op = FTDNN_conv_model(
        window_size,
        exog_dim
    )

    model = Model(
        inputs=model_in,
        outputs=model_op
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # Training Data
    train_x_exog, train_x_end, train_y = ctdnn_get_data(window_size, 'train')
    x_inputs_train = train_x_exog
    x_inputs_train.append(train_x_end)

    # model.summary()
    h = model.fit(x=x_inputs_train,
                  y=train_y,
                  epochs=epochs,
                  batch_size=batch_size)
    train_loss = h.history['loss'][-1]



    test_x_exog, test_x_end, test_y = ctdnn_get_data(window_size, 'test')
    x_inputs_test = test_x_exog
    x_inputs_test.append(test_x_end)

    test_mse = model.evaluate(x_inputs_test, test_y, batch_size=batch_size)
    return train_loss,test_mse

def experiment():
    global window_size
    # Vary window sizes !
    res_dict = {}

    for w in [32, 64, 128, 512]:
        window_size = w
        ctdnn_get_data(window_size)
        train_loss, test_mse =  FTDNN_conv()
        res_dict[w] = [train_loss, test_mse]

    print 'Result, for varying window sizes'
    print res_dict
    return

experiment()