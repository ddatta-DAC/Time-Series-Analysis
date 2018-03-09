import numpy as np
from keras.models import load_model
from keras.layers import Dense
import h5py
import keras
import data_feeder
from itertools import tee, izip


def get_stacked_ae():
    nb_epoch = 2000
    batch_size = 256
    X_train, _, _, _, _ = data_feeder.get_data(True)
    num_layers = 3
    shape = [64, 32, 16]
    inp_dim = X_train.shape[-1]

    model = keras.models.Sequential()

    # input_layer = keras.layers.Dense(units=inp_dim,
    #                                  input_dim=inp_dim,
    #                                  activation=None,
    #                                  use_bias=False,
    #                                  activity_regularizer=None)
    # model.add(input_layer)

    for i in range(num_layers):
        if i == 0:
            layer = Dense(shape[i], input_dim=inp_dim, activation='sigmoid')
        else:
            layer = Dense(shape[i], activation='tanh')

        model.add(layer)
        layer = Dense(units=inp_dim, activation='tanh')
        model.add(layer)

        print model.summary()
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.MSE,
                      metrics=['accuracy'])

        model.fit(X_train,
                  X_train,
                  epochs=nb_epoch,
                  batch_size=batch_size,
                  shuffle=False)

        for layer in model.layers:
            layer.trainable = False

        model.pop()
        print model.summary()
    model.save('ae_model.h5')
    return model


# get_stacked_ae()

def get_windowed_inp(_arr, window_size):
    _arr = list(np.reshape(_arr, [_arr.shape[0]]))

    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)

    inp = []
    op = []
    for w in window(_arr, window_size+1):
        inp.append(w[:-1])
        op.append(w[-1])

    inp = np.asarray(inp)
    op = np.asarray(op)
    print inp.shape
    print op.shape

    return inp, op


def create_complete_model():
    # model = load_model('ae_model.h5')
    model = keras.models.Sequential()

    X_train, X_test, Y_train, Y_test, scaler_array = data_feeder.get_data()

    window_size = 8
    # create windows
    get_windowed_inp(Y_train, window_size)
    return

    units = 64

    batch_input_shape = (batch_size, prev_step, 1)
    lstm1 = keras.layers.LSTM(128, use_bias=True,
                              batch_input_shape=batch_input_shape,
                              dropout=0.20,
                              stateful=True,
                              return_sequences=True,
                              )
    model.add(lstm1)
    lstm2 = keras.layers.LSTM(64, use_bias=True,
                              dropout=0.35,
                              stateful=True,
                              return_sequences=True,
                              )
    model.add(lstm2)
    print model.summary()


create_complete_model()
