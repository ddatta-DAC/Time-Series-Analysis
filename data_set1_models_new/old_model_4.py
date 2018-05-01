import keras.layers as layers
import keras.models as models
import keras
import numpy as np
import data_feeder
import utils
import math


# --------------------------- #
def get_training_data(time_window, endog_only=True):
    # window of training data = time_window
    _, _, z, _, _ = data_feeder.get_data(std=True)

    if endog_only == True:
        z = np.asanyarray(z)
        print '>>', z.shape
        z = np.reshape(z, [-1, 1])

        print z.shape
        res = utils.get_windowed_data(z, time_window)
        print ' >>> ', res.shape
        _x = res[:, -1:]
        _y = res[:, 0:time_window]
        print ' >>> ', _x.shape, _y.shape
        _x = np.reshape(_x, [_x.shape[0], 1, 1])
        _y = np.reshape(_y, [_y.shape[0], time_window, 1])
        return _y, _x
    else:
        pass

    return


# --------------------------- #

class model:

    def __init__(self):
        self.model = None
        self.save_file = 'model_4.h5'
        return

    def set_hyperparameters(self, batch_size, epochs, time_window, num_layers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_window = time_window
        self.num_layers = num_layers
        return

    def set_train_data(self, train_x=None, train_y=None):
        # train_x = np.random.rand(1000, self.time_window, 1)
        # train_y = np.random.rand(1000, 1)
        self.train_x = train_x
        self.train_y = train_y
        return

    def res_block(self, inp, num_filters, dilation, kernel_size):
        # down-sampling is performed with a stride of 2
        conv_layer_op = layers.Conv1D(filters=num_filters,
                                      kernel_size=kernel_size,
                                      dilation_rate=dilation,
                                      strides=1,
                                      kernel_regularizer=keras.regularizers.l2(0.01),
                                      padding='valid')(inp)
        print ' shape :', conv_layer_op.shape
        res = layers.BatchNormalization()(conv_layer_op)
        res = layers.LeakyReLU()(res)

        # res = layers.add([res, inp])
        return res

    def build(self):
        # input is an image of time_window x 1
        input_layer = layers.Input(shape=(self.time_window, 1))

        inp = input_layer

        for l in range(self.num_layers):
            print '---'
            d = math.pow(2, l)
            print 'layer', l, 'dialtion rate :', d
            network_op = self.res_block(
                inp=inp,
                num_filters=4,
                dilation=[d],
                kernel_size=2
            )

            inp = network_op

        network_op = layers.Conv1D(filters=1,
                                   kernel_size=1,
                                   dilation_rate=[1],
                                   strides=1,
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   padding='valid')(inp)

        model = models.Model(
            inputs=[input_layer],
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
        for _ in range( self.epochs ):
            _x = self.train_x
            _y = self.train_y
            bs = self.batch_size

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
            print hist.history['loss']
        return self.train_mse, self.train_mae


num_layers = 4
time_steps = int(math.pow(2, num_layers))
print 'time_steps ', time_steps
train_x, train_y = get_training_data(time_steps)
model_obj = model()
model_obj.set_hyperparameters(epochs=200, batch_size=128, time_window=time_steps, num_layers=num_layers)
model_obj.set_train_data(train_x, train_y)
model_obj.build()
plot_list = model_obj.train_model()
# ---------------- #

op_file = 'model_4.txt'
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
