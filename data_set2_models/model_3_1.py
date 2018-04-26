import keras
import os
import numpy as np
from keras.layers import Dense
import data_feeder_2 as data_feeder
from keras.models import load_model
from itertools import tee, izip
import utils

ae_1_epoch = 500
ae_2_epoch = 500
lstm_epochs = 500
lstm_time_steps = 16
ae_1_units = [64, 32, 16]
ae_2_units = [64, 32, 16]
lstm_units = [32, 16]
lstm_batch_size = 128

# set up autoencoder with inputs as [ x(t-k)...x(t-1), y(t) ]
exog_lag = 32
endo_lag = 8


# window the exogenous input
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


# stacked autoencoder
class ae_class:

    def __init__(self, name=''):
        self.save_file = 'model_3_1_ae_obj_' + str(name) + '.h5'
        self.train_losses = []
        self.train_acc = []
        return

    def set_data(self, data_x):
        self.X = data_x
        return

    def load_model(self):
        if os.path.exists(self.save_file):
            self.model = load_model(self.save_file)
        else:
            return self.build_model()
        return None

    def set_hyperparams(self, layer_units, inp_dim, batch_size, epochs):
        self.epoch = epochs
        self.batch_size = batch_size
        self.num_layers = len(layer_units)
        self.num_units = layer_units  # [64, 32, 24]
        self.inp_dim = inp_dim

    def build_model(self):

        model = keras.models.Sequential()
        for i in range(self.num_layers):
            self.train_losses = []
            self.train_acc = []

            if i == 0:
                op_x = self.X
            else:
                op_x = model.predict(self.X)

            # encoder layer
            if i == 0:
                # first layer
                layer = Dense(self.num_units[i], input_dim=self.inp_dim, activation='tanh')
            else:
                layer = Dense(self.num_units[i], activation='tanh')

            model.add(layer)
            # decoder

            if i == 0:
                dec_units = self.inp_dim
            else:
                dec_units = self.num_units[i - 1]

            layer = Dense(units=dec_units, activation='tanh')
            model.add(layer)
            model.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.MSE,
                          metrics=['mse'])
            # ===== #
            # Train the model

            # create batches from the data :
            x = self.X
            y = op_x
            num_batches = (x.shape[0] - 1) // self.batch_size + 1
            print ' Auto encoder .. number of batches in training :', num_batches, ' batch size :', self.batch_size
            for _ in range(self.epoch):

                for i in range(num_batches):
                    _x = x[i * self.batch_size:(i + 1) * self.batch_size, :]
                    _y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
                    bs = _x.shape[0]
                    hist = model.fit(
                        _x,
                        _y,
                        epochs=5,
                        batch_size=bs,
                        shuffle=False,
                        verbose=False
                    )
                    self.train_losses.extend(hist.history['loss'])
            for layer in model.layers:
                layer.trainable = False

            model.pop()

        print model.summary()
        model.save(self.save_file)
        self.model = model
        return self.train_losses


# ------ #
# Get Data
# ------ #
def get_ae_1():
    X_train, _, Y_train, _, _ = data_feeder.get_data(True)

    # set up autoencoder for exogenous values
    ae_obj_1 = ae_class(1)

    ae_obj_1.set_hyperparams(
        layer_units=ae_1_units,
        inp_dim=X_train.shape[-1],
        batch_size=128,
        epochs=ae_1_epoch
    )

    ae_obj_1.set_data(X_train)
    ae_1_losses = ae_obj_1.load_model()
    return ae_obj_1


# print 'Shape of training data for 1st auto encoder', X_train.shape
# print 'Auto Encoder 1'
# ae_obj_1.model.summary()
# ------ #

def get_ae_2():
    global exog_lag
    global endo_lag
    ae_obj_1 = get_ae_1()

    X_train, _, Y_train, _, _ = data_feeder.get_data(True)
    xformed_exog_train = ae_obj_1.model.predict(X_train)
    _x1 = get_windowed_data(data=xformed_exog_train, window_size=exog_lag)
    _y1 = get_windowed_data(data=Y_train, window_size=endo_lag)

    num_samples = min(_x1.shape[0], _y1.shape[0])
    _x1 = _x1[-num_samples:]
    _y1 = _y1[-num_samples:]
    ae2_train_data = np.concatenate([_x1, _y1], axis=-1)
    ae_2_inp_dim = ae2_train_data.shape[-1]

    name = str(2) + '_' + str(exog_lag) + '_' + str(endo_lag)

    ae_obj_2 = ae_class(name)
    ae_obj_2.set_hyperparams(
        layer_units=ae_2_units,
        inp_dim=ae_2_inp_dim,
        batch_size=128,
        epochs=ae_2_epoch
    )
    ae_obj_2.set_data(ae2_train_data)
    ae_2_losses = ae_obj_2.load_model()

    return ae_obj_1 , ae_obj_2

    # ---------- #
    # print 'Auto Encoder 2'
    # ae_obj_2.model.summary()
    # ---------- #


# Use the output of the hierarchical AE to be  input for a LSTM

class lstm_model:

    def __init__(self, name):
        self.model = None
        self.save_file = 'model_3_1_lstm_' + name + '.h5'
        self.train_losses = []
        self.train_acc = []
        return

    def set_hyperparams(self, lstm_units, time_step, epochs, batch_size):

        self.time_step = time_step
        self.inp_dimension = None
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        return

    def load_model(self):
        if os.path.exists(self.save_file):
            self.model = load_model(self.save_file)
            return None
        else:
            return self.build_model()

    def set_train_data(self, x, y):
        self.x = x
        self.y = y
        # self.x = np.reshape(self.x, [-1,self.time_step,self.x.shape[-1]])
        # self.y = np.reshape(self.y, [-1, self.time_step, self.y.shape[-1]])
        self.inp_dimension = self.x.shape[-1]

    def build_model(self):

        self.train_losses = []
        self.train_mape = []
        self.train_mse = []
        self.train_mae = []

        model = keras.models.Sequential()
        layer1 = keras.layers.LSTM(
            units=self.lstm_units[0],
            input_shape=(self.time_step, self.inp_dimension),
            activation='tanh',
            return_sequences=True
        )
        model.add(layer1)
        layer2 = keras.layers.LSTM(
            units=self.lstm_units[1],
            activation='tanh',
            return_sequences=True
        )
        model.add(layer2)
        layer3 = keras.layers.TimeDistributed(keras.layers.Dense(units=16, activation='tanh'))
        model.add(layer3)
        layer4 = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='tanh'))
        model.add(layer4)
        model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MSE,
            metrics=[keras.metrics.mse]
        )

        self.model = model
        model.summary()
        self.train_model()
        return self.train_losses

    def train_model(self):

        if os.path.exists(self.save_file):
            return

        self.error_keys = {
            'mse': 'mean_squared_error',
            'mae': 'mean_absolute_error',
            'mape': 'mean_absolute_percentage_error'
        }

        batch_size = self.batch_size * self.time_step
        num_batches = (self.x.shape[0] - 1) // batch_size + 1

        self.train_losses = []
        self.train_mape = []
        self.train_mse = []
        self.train_mae = []

        for _ in range(self.epochs):

            for i in range(num_batches):
                _x = self.x[i * batch_size:(i + 1) * batch_size, :]
                _y = self.y[i * batch_size:(i + 1) * batch_size, :]
                if _x.shape[0] != batch_size:
                    _samples = _x.shape[0] - (_x.shape[0] % (self.time_step * _x.shape[-1]))
                    _x = _x[-_samples:, :]
                    _y = _y[-_samples:, :]

                bs = _x.shape[0] // self.time_step

                _x = np.reshape(_x, [bs, self.time_step, _x.shape[-1]])
                _y = np.reshape(_y, [bs, self.time_step, _y.shape[-1]])

                hist = self.model.fit(
                    _x,
                    _y,
                    epochs=2,
                    batch_size=bs,
                    shuffle=False,
                    verbose=False
                )

                self.train_losses.extend(hist.history['loss'])
                self.train_mse.extend(hist.history[self.error_keys['mse']])

        self.model.save(self.save_file)

    def test(self, x, y):
        print 'In test ...'

        _samples = self.time_step * (x.shape[0] // self.time_step)
        x = x[-_samples:, :]
        y = y[-_samples:, :]

        s_n = _samples / self.time_step
        x = np.reshape(x, [-1, self.time_step, x.shape[-1]])
        y = np.reshape(y, [s_n, self.time_step, 1])

        result = self.model.evaluate(x, y)
        print 'Mean Square Error', result
        return result[0]


# Input to lstm should be output of the auto-encoder layer of d-dimension
# format : [samples, timestep , d]
# Output should be of the format
# [ samples, timesteps , 1]


def build_trian_model():

    global endo_lag
    global exog_lag
    global lstm_units
    global lstm_batch_size

    X_train, _, Y_train, _, _ = data_feeder.get_data(True)

    ae_1_obj, ae_2_obj = get_ae_2()
    xformed_exog_train = ae_1_obj.model.predict(X_train)

    _x1 = get_windowed_data(data=xformed_exog_train, window_size=exog_lag)
    _x2 = get_windowed_data(data=Y_train, window_size=endo_lag + 1)
    _x2 = _x2[:, 0:endo_lag]
    _y = _x2[:, -1:]

    num_samples = min(_x1.shape[0], _x2.shape[0])
    _x1 = _x1[-num_samples:]
    _x2 = _x2[-num_samples:]
    _y = _y[-num_samples:, :]

    ae2_inp = np.concatenate([_x1, _x2], axis=-1)
    x = ae_2_obj.model.predict(ae2_inp)

    name = str(exog_lag) + '_' + str(endo_lag)
    lstm_model_obj = lstm_model(name)
    lstm_model_obj.set_hyperparams(
        lstm_units=lstm_units,
        time_step=lstm_time_steps,
        epochs=lstm_epochs,
        batch_size=lstm_batch_size
    )
    lstm_model_obj.set_train_data(x, _y)
    lstm_model_obj.load_model()
    return ae_1_obj, ae_2_obj, lstm_model_obj


# Test
# ae_obj_1 = ae_class(1)
# ae_obj_2 = ae_class(2)


def experiment():

    ae_1_obj, ae_2_obj, lstm_model_obj = build_trian_model()

    # Test
    _, X_test, _, Y_test, _ = data_feeder.get_data(True)
    xformed_exog_test = ae_1_obj.model.predict(X_test)

    _x1 = get_windowed_data (
        data=xformed_exog_test,
        window_size=exog_lag
    )
    _x2 = get_windowed_data(
        data=Y_test,
        window_size=endo_lag + 1
    )
    _x2 = _x2[:, 0:endo_lag]
    _y = _x2[:, -1:]
    test_y = _y


    num_samples = min(_x1.shape[0], _x2.shape[0])
    _x1 = _x1[-num_samples:]
    _x2 = _x2[-num_samples:]
    test_y = test_y[-num_samples:, :]

    ae2_inp = np.concatenate([_x1, _x2], axis=-1)
    test_x = ae_2_obj.model.predict(ae2_inp)
    test_mse = lstm_model_obj.test(test_x, test_y)
    return test_mse


def run_all():
    global exog_lag
    global endo_lag

    res_dict = {}
    for ex in [8, 32, 64]:
        res_dict[ex] = {}
        for en in [8, 32, 64] :
            exog_lag = ex
            endo_lag = en
            res = experiment()
            res_dict[ex][en] = res

    print 'Test MSE'
    print res_dict
    return

run_all()

