import keras.layers as layers
import keras.models as models
import keras
import numpy as np
import data_feeder
import utils


class model:

    def __init__(self):
        self.model = None
        self.build()
        return

    def set_hyperparameters(self):
        self.batch_size = 128
        self.time_window = 25
        return


    def set_train_data(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        return


    def res_block(self, inp, num_filters, kernel_size):
        # down-sampling is performed with a stride of 2
        conv_layer_op = layers.Conv1D(filters=num_filters,
                                      kernel_size=kernel_size,
                                      strides=[1],
                                      padding='same')(inp)

        res = layers.BatchNormalization()(conv_layer_op)
        res = layers.LeakyReLU()(res)
        res = layers.add([res, inp])
        return res


    def build(self):
        inp_data = np.random.rand(10, 10, 1)
        # input is an image of 10 x 1

        input_layer = layers.Input(shape=(10, 1))
        network_op = self.res_block(
            inp=input_layer,
            num_filters=4,
            kernel_size=[3]
        )
        model = models.Model(
            input = [input_layer],
            outputs = [network_op]
        )
        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = keras.losses.MSE,
            metrics = [keras.metrics.mae, keras.metrics.mse]
        )

        print model.summary()
        return


    def train_model(self):
        self.error_keys = {
            'mse': 'mean_squared_error',
            'mae': 'mean_absolute_error',
        }
        self.train_losses = []
        self.train_mape = []
        self.train_mse = []
        self.train_mae = []
        _x = self.train_x
        _y = self.train_y
        bs =100

        hist = self.model.fit(
            _x,
            _y,
            epochs=10,
            batch_size=bs,
            shuffle=False,
            verbose=False
        )

        self.train_losses.extend(hist.history['loss'])
        self.train_mse.extend(hist.history[self.error_keys['mse']])
        self.train_mae.extend(hist.history[self.error_keys['mae']])
        return


model()

# ---------------- #
