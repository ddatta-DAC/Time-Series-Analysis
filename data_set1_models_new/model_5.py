# basic encoder decoder model
import keras
import keras.layers as layers
import keras.models as models
import numpy as np
from itertools import tee, izip
import data_feeder
import os
from keras.models import load_model
import pprint


def get_windowed_data(data, window_size):
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

# ------------------------------ #
exog_dim  = 5
encoder_units = [16,8]
decoder_units = [16,8]

# ------------------------------ #

class encoder_decoder_model:

    def __init__(self):
        self.model_file = 'model_5_obj.h5'
        self.model = None
        return

    def set_hyperparams(
            self,
            exg_dimension,
            end_window_size,
            encoder_layer_units,
            decoder_layer_units,
            epochs,
            lstm_time_steps,
            batch_size
    ):

        self.exg_dimension = exg_dimension
        self.encoder_layer_units = encoder_layer_units
        self.decoder_layer_units = decoder_layer_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_time_steps = lstm_time_steps
        self.end_window_size = end_window_size
        return

    def build_model(self):

        if os.path.exists(self.model_file):
            pass
            # self.model = load_model(self.model_file)
            # return

        # First input layer for the Network
        input_layer_1 = layers.Input(
            shape=(self.lstm_time_steps, self.exg_dimension)
        )
        current_op = input_layer_1
        num_encoder_layers = len(self.encoder_layer_units)
        for l in range(num_encoder_layers):
            if l == 0:
                inp = input_layer_1
                lstm_inp_dim = self.exg_dimension


            op = layers.LSTM(
                units=self.encoder_layer_units[l],
                input_shape=(
                    self.lstm_time_steps,
                    lstm_inp_dim
                ),
                activation='tanh',
                return_sequences=True
            )(inp)

            inp = op
            # For next layer
            lstm_inp_dim = self.encoder_layer_units[l]
            current_op = op

        # Input 2 should be be of the shape [ None, lstm_time_step , end__window_size ]
        input_layer_2 = layers.Input(shape=(self.lstm_time_steps, self.end_window_size))
        current_op = layers.concatenate([current_op, input_layer_2], axis=-1)
        print current_op

        num_decoder_layers = len(self.decoder_layer_units)
        lstm_inp_dim = current_op.shape[-1]

        for l in range(num_decoder_layers):
            if l == 0:
                inp = current_op

            op = layers.LSTM(
                units=self.encoder_layer_units[l],
                input_shape=(self.lstm_time_steps, lstm_inp_dim),
                activation='tanh',
                return_sequences=True
            )(inp)

            inp = op
            lstm_inp_dim = self.encoder_layer_units[l]
            current_op = op

        print current_op

        # Single output for each time step is single output
        final_op = layers.TimeDistributed(layers.Dense(units=1))(current_op)

        print final_op

        model = models.Model(
            inputs=[input_layer_1, input_layer_2],
            outputs=[final_op]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MSE,
            metrics=[keras.metrics.mse]
        )

        self.model = model
        self.model.summary()
        return

    def format_data_xy(self, X, Y):

        x_end_windowed = get_windowed_data(Y, self.end_window_size + 1)
        x2 = x_end_windowed[:, 0: self.end_window_size]
        y = x_end_windowed[:, -1:]

        # Reshape both x and y to be :
        # num_samples , time_step , dimension

        num_samples = min(X.shape[0], y.shape[0])
        num_samples = self.lstm_time_steps * (num_samples // self.lstm_time_steps)
        X = X[-num_samples:, :]
        print 'num_samples', num_samples

        x1 = np.reshape(X, [-1, self.lstm_time_steps, self.exg_dimension])
        x2 = x2[-num_samples:, :]
        x2 = np.reshape(x2, [-1, self.lstm_time_steps, x2.shape[-1]])

        y = y[-num_samples:, :]
        y = np.reshape(y, [-1, self.lstm_time_steps, 1])

        print 'Shape of x1 ', x1.shape
        print 'Shape of x2', x2.shape
        print 'Shape of y', y.shape

        return x1, x2, y

    def train_model(self):

        exog_train, _, _, end_train, _, _ , _ = data_feeder.get_data_val(True)
        x1, x2, y = self.format_data_xy(exog_train, end_train)
        hist = self.model.fit([x1,x2],y,epochs=self.epochs)
        trian_loss = hist.history['loss']
        return np.mean(trian_loss)


    def val_model(self):
        _, exog_val, _,  _, end_val, _, _ = data_feeder.get_data_val(True)
        x1, x2, y = self.format_data_xy(exog_val, end_val)
        score = self.model.evaluate([x1,x2],y)
        val_mse = score[0]
        return val_mse


    def test_model(self):

        _, _, exog_test, _, _, end_test, _ = data_feeder.get_data_val(True)
        x1, x2, y = self.format_data_xy(exog_test, end_test)
        score = self.model.evaluate([x1,x2],y)
        test_mse = score[0]
        return test_mse

# ------------------------------- #

import pandas as pd

def experiment():

    exg_dimension = 81
    end_window_size = 16
    encoder_layer_units = [16, 8]
    decoder_layer_units = [16, 8]
    epochs = 350
    lstm_time_steps = 16
    batch_size = 128

    _enc_units = [[16, 8], [16, 16], [32, 16]]
    _dec_units = [[16, 8], [16, 16], [32, 16]]
    ts = [16, 32, 64]
    columns = ['End Window Size',
               'Encoder Layer 1 Units',
               'Encoder Layer 2 Units',
               'Decoder Layer 1 Units',
               'Decoder Layer 2 Units',
               'Train Error',
               'Validation Error',
               'Test Error'
               ]
    df = pd.DataFrame(columns=columns)

    for t in ts :

        end_window_size = t

        for enc in _enc_units :
            for dec in _dec_units :
                res_dict = {}
                model_obj = encoder_decoder_model()
                model_obj.set_hyperparams(
                    exg_dimension,
                    end_window_size,
                    encoder_layer_units,
                    decoder_layer_units,
                    epochs,
                    lstm_time_steps,
                    batch_size
                )

                model_obj.build_model()
                train_mse = model_obj.train_model()
                val_mse = model_obj.val_model()
                test_mse = model_obj.test_model()

                res_dict[columns[0]] = t

                res_dict[columns[1]] = enc[0]
                res_dict[columns[2]] = enc[1]
                res_dict[columns[3]] = dec[0]
                res_dict[columns[4]] = dec[1]

                res_dict[columns[5]] = train_mse
                res_dict[columns[6]] = val_mse
                res_dict[columns[7]] = test_mse
                df = df.append(res_dict, ignore_index=True)
    df.to_csv('model_5_op.csv')

experiment()