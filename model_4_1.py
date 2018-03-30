import keras.layers as layers
import keras.models as models
import keras
import numpy as np
import data_feeder
import utils
import math
from keras.models import load_model
import tensorflow as tf
from keras.layers.core import Reshape


# --------------------------- #
def get_training_data(time_window):
    # window of training data = time_window
    # X : exogenous series
    # Y : target series

    exog_train, exog_test, end_train, end_test, _ = data_feeder.get_data(std=True)
    exog_dim = exog_train.shape[-1]
    res = utils.get_windowed_data(exog_train, time_window)
    res = np.reshape(res, [-1, time_window, 1, exog_dim])
    # time_window, 1, exog_dim
    exog_train_x = res

    res = utils.get_windowed_data(end_train, time_window)
    res = np.reshape(res, [-1, time_window, 1])
    end_train_x = res

    n_samples = res.shape[0]
    res = utils.get_windowed_data(end_train, 1)
    res = res[-n_samples:, :]
    res = np.reshape(res,[-1,1,1])
    end_train_y = res

    return exog_train_x, end_train_x, end_train_y


# --------------------------- #

class model:

    def __init__(self):
        self.model = None
        self.save_file = 'model_4_1.h5'
        return

    def set_hyperparameters(self, batch_size, epochs, time_window, parallel_layers, comb_layers, exg_red_layers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_window = time_window
        self.num_layers = num_layers
        self.exog_dim = 81
        self.parallel_layers = parallel_layers
        self.comb_layers = comb_layers
        self.exg_red_layers = exg_red_layers

        return

    def set_train_data(self, exog_train_x, end_train_x, end_train_y):

        self.exog_train_x = exog_train_x
        self.end_train_x = end_train_x
        self.end_train_y = end_train_y
        return

    def res_block(self, inp, num_filters, dilation, kernel_size):

        conv_layer_op = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            strides=1,
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid'
        )(inp)
        print ' shape :', conv_layer_op.shape
        res = layers.BatchNormalization()(conv_layer_op)
        res = layers.LeakyReLU()(res)

        # res = layers.add([res, inp])
        return res

    def build_exog_cnn(self, input_exg):
        # exogenous input is an image of time_window x 1 x channels[=exog_dim]
        exg_op = input_exg
        n_f = int(math.pow(4, self.exg_red_layers - 1))
        for l in range(self.exg_red_layers):
            exg_op = layers.Conv2D(
                filters=n_f,
                kernel_size=[1, 1],
                dilation_rate=[1, 1],
                strides=[1, 1],
                kernel_regularizer=keras.regularizers.l2(0.01),
                padding='same'
            )(exg_op)
            exg_op = layers.BatchNormalization()(exg_op)
            exg_op = layers.LeakyReLU()(exg_op)

            n_f /= 4

        w = exg_op.shape.as_list()[1]
        exg_op = Reshape((w, 1))(exg_op)

        return exg_op

    def build(self):

        # input is an image of time_window x 1
        input_exg = layers.Input(shape=(self.time_window, 1, self.exog_dim))
        input_end = layers.Input(shape=(self.time_window, 1))

        op_end = input_end
        op_exg = self.build_exog_cnn(input_exg)

        print '---> Start of parallel layers'
        # build parallel dilated layers
        for l in range(self.parallel_layers):
            print '--- parallel layer  ', l + 1
            d = math.pow(2, l)
            print 'dialtion rate :', d

            op_end = self.res_block(
                inp=op_end,
                num_filters=4,
                dilation=[d],
                kernel_size=2
            )

            op_exg = self.res_block(
                inp=op_exg,
                num_filters=4,
                dilation=[d],
                kernel_size=2
            )

        # build combined dilated layers
        print '---> Start of combined layers'
        comb_op = layers.add([op_exg, op_end])
        print comb_op

        for l in range(self.parallel_layers, self.parallel_layers + self.comb_layers):

            d = math.pow(2, l)
            print 'layer', l + 1, 'dialtion rate :', d

            comb_op = self.res_block(
                inp=comb_op,
                num_filters=4,
                dilation=[d],
                kernel_size=2
            )

        network_op = layers.Conv1D(filters=1,
                                   kernel_size=1,
                                   dilation_rate=[1],
                                   strides=1,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   padding='valid')(comb_op)
        network_op = layers.LeakyReLU()(network_op)

        model = models.Model(
            inputs=[input_exg, input_end],
            outputs=[network_op]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MSE,
            metrics=[keras.metrics.mae, keras.metrics.mse]
        )

        print model.summary()
        self.model = model
        return

    def train_model(self):

        self.error_keys = {
            'mse': 'mean_squared_error',
            'mae': 'mean_absolute_error',
        }

        self.train_losses = []
        self.train_mse = []
        self.train_mae = []

        for _ in range(self.epochs):

            bs = self.batch_size
            hist = self.model.fit(
                x = [self.exog_train_x,self.end_train_x],
                y = self.end_train_y,
                epochs=10,
                batch_size=bs,
                shuffle=False,
                verbose=False
            )

            self.train_losses.extend(hist.history['loss'])
            self.train_mse.extend(hist.history[self.error_keys['mse']])
            self.train_mae.extend(hist.history[self.error_keys['mae']])
            print hist.history['loss']
        return self.train_mse, self.train_mae


# -------------------- #

exg_red_layers = 3
parallel_layers = 2
comb_layers = 3
num_layers = parallel_layers + comb_layers

time_steps = int(math.pow(2, num_layers))
exog_train_x, end_train_x, end_train_y = get_training_data(time_steps)
model_obj = model()

model_obj.set_hyperparameters(
    epochs=200,
    batch_size=128,
    time_window=time_steps,
    parallel_layers=parallel_layers,
    comb_layers=comb_layers,
    exg_red_layers=exg_red_layers
)
model_obj.set_train_data(exog_train_x, end_train_x, end_train_y)
model_obj.build()
plot_list = model_obj.train_model()

op_file = 'model_4_1.txt'
fp = open(op_file, 'w')

for z in plot_list:
    fp.write(str(z))
    fp.write('\n')
fp.close()

for z in plot_list:
    if z is None: continue
    print np.asanyarray(z).shape
    xz = range(len(z))
    utils.plot(xz, z)


