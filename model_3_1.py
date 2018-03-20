import keras
import numpy as np
from keras import Model
from keras.layers import Dense
import data_feeder
from keras.models import load_model
from itertools import tee, izip


# stacked autoencoder
class ae_class:

    def __init__(self, name=''):
        self.save_file = 'ae_obj_' + str(name) + '.h5'
        return

    def set_data(self, data_x):
        self.X = data_x
        return

    def load_model(self):
        self.model = load_model(self.save_file)

    def set_hyperparams(self, layer_units, batch_size, epochs):
        self.epoch = epochs
        self.batch_size = batch_size
        self.num_layers = len(layer_units)
        self.num_units = layer_units  # [64, 32, 24]
        self.inp_dim = X_train.shape[-1]

    def build_model(self):

        model = keras.models.Sequential()
        for i in range(self.num_layers):
            print 'Building layer ', i + 1

            if i == 0:
                op_x = X_train
            else:
                op_x = model.predict(X_train)

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

            print 'dec units', dec_units

            layer = Dense(units=dec_units, activation='tanh')
            model.add(layer)
            model.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.MSE,
                          metrics=['accuracy'])

            model.fit(
                X_train,
                op_x,
                epochs=self.epoch,
                batch_size=self.batch_size,
                shuffle=False,
                verbose=True
            )

            for layer in model.layers:
                layer.trainable = False

            model.pop()

        print model.summary()
        model.save(self.save_file)
        self.model = model


# window the exogenous input
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
        w = np.reshape(w,[-1])
        op.append(w)

    op = np.asarray(op)
    return op


X_train, _, Y_train, _, _ = data_feeder.get_data(True)
# set up autoencoder for exogenous values
ae_obj_1 = ae_class(1)
ae_obj_1.set_hyperparams(layer_units=[64, 32, 16], batch_size=512, epochs=5000)
ae_obj_1.set_data(X_train)
ae_obj_1.load_model()
print ae_obj_1.model.summary()

# set up autoencoder with inputs as [ x(t-k)...x(t-1), y(t) ]
exog_lag = 10
endo_lag = 5
xformed_exog_train = ae_obj_1.model.predict(X_train)
_x1 = get_windowed_data_ex(data=xformed_exog_train, window_size=exog_lag)
_y1 = get_windowed_data_ex(data=Y_train, window_size=endo_lag)

print _x1.shape
print _y1.shape
l = min(_x1.shape[0],_y1.shape[0])
_x1 = _x1[-l:]
_y1 = _y1[-l:]
ae2_train_data = np.concatenate([_x1,_y1],axis=-1)
print _x1.shape
print _y1.shape

ae_obj_2 = ae_class(2)
ae_obj_2.set_hyperparams(layer_units=[64, 32, 16], batch_size=512, epochs=5000)
ae_obj_2.set_data(ae2_train_data)
ae_obj_2.build_model()
print ae_obj_2.model.summary()
