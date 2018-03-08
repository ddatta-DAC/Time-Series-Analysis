import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.datasets import mnist
import numpy as np

import numpy as np
import keras
import data_feeder


def get_stacked_ae():
    nb_epoch = 200
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

    return model

get_stacked_ae()

def create_complete_model():
    model = get_stacked_ae()

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