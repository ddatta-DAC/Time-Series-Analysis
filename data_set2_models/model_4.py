import keras.layers as layers
import keras.models as models
import keras
import numpy as np
import utils
import math
import os
import sys
import data_feeder_2 as data_feeder
import pandas as pd

# --------------------------- #

def get_data(time_window, type ):

    # window of training data = time_window
    X_train,_,  X_test, Y_train,Y_val, Y_test, _ = data_feeder.get_data_val(std=True)

    if type == 'test':
        y = Y_test
    elif type == 'val':
        y = Y_val
    else:
        y = Y_train

    y = np.asanyarray(y)
    y = np.reshape(y, [-1, 1])
    res = utils.get_windowed_data(y, time_window + 1)
    x = res[:, 0:time_window]
    y = res[:, -1:]
    x = np.reshape(x, [x.shape[0], time_window, 1])
    y = np.reshape(y, [y.shape[0], 1])
    return x, y



def get_test_data(time_window):
    type = "test"
    return get_data(time_window, type )

def get_training_data(time_window):
    type = "train"
    return get_data(time_window, type)

def get_val_data(time_window):
    type = "val"
    return get_data(time_window, type)


# --------------------------- #

class model:

    def __init__(self):
        self.model = None
        return

    def set_file_name(self):
        self.save_file = 'model_4' + str(self.time_window) + '.h5'

    def set_hyperparameters(self, batch_size, epochs, time_window, num_layers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_window = time_window
        self.num_layers = num_layers
        self.set_file_name()
        return

    def set_train_data(self, train_x=None, train_y=None):
        self.train_x = train_x
        self.train_y = train_y
        return

    def res_block(self, inp, num_filters, dilation, kernel_size):
        # down-sampling is performed with a stride of 2
        conv_layer_op = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            strides=1,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid'
        )(inp)


        res = layers.BatchNormalization()(conv_layer_op)
        res = layers.LeakyReLU()(res)
        return res

    def build(self):
        # input is an image of time_window x 1

        input_layer = layers.Input(shape=(self.time_window, 1))
        inp = input_layer
        network_op = None

        for l in range(self.num_layers):
            d = math.pow(2, l)
            network_op = self.res_block(
                inp=inp,
                num_filters = 4,
                dilation=[d],
                kernel_size=2
            )

            inp = network_op

        network_op = layers.Conv1D(filters=1,
                                   kernel_size=1,
                                   dilation_rate=[1],
                                   strides=1,
                                   activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(0.01),
                                   padding='valid')(inp)

        network_op = layers.Flatten()(network_op)

        model = models.Model(
            inputs=[input_layer],
            outputs=[network_op]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MSE,
            metrics=[keras.metrics.mae, keras.metrics.mse]
        )

        self.model = model
        return


    def train_model(self):
        self.error_keys = {
            'mse': 'mean_squared_error'
        }

        train_loss = []
        for _ in range( self.epochs ):
            _x = self.train_x
            _y = self.train_y
            bs = self.batch_size

            hist = self.model.fit(
                _x,
                _y,
                epochs=5,
                batch_size=bs,
                shuffle=False,
                verbose=False
            )


            train_loss.extend(hist.history['loss'])
        train_loss = np.mean(train_loss)
        print 'Train Loss',train_loss
        return train_loss

    def val_model(self,val_x,val_y):
        score = self.model.evaluate(
            val_x,
            val_y,
            batch_size=1
        )

        print 'Validation Score', np.mean(score)
        return np.mean(score)

    def test_model(self, test_x, test_y):

        score = self.model.evaluate(
            test_x,
            test_y,
            batch_size=1
        )
        print 'Test Score', np.mean(score)
        return np.mean(score)

# ---------------------- #
def experiment() :
    _num_layers = [ 3,4,5,6 ]

    columns = [
        'Num Layers',
        'Time Window',
        'Train Error',
        'Validation Error',
        'Test Error'
    ]
    df = pd.DataFrame(columns=columns)

    batch_size = 64
    epochs = 175

    for num_layers in _num_layers:
        res_dict = {}

        time_steps = int(math.pow(2, num_layers))
        print 'time_steps ', time_steps

        train_x, train_y = get_training_data(time_steps)
        val_x, val_y = get_val_data(time_steps)
        test_x, test_y = get_test_data(time_steps)

        model_obj = model()
        model_obj.set_hyperparameters(
            epochs = epochs,
            batch_size = batch_size,
            time_window = time_steps,
            num_layers = num_layers
        )

        model_obj.build()
        model_obj.set_train_data(
            train_x,
            train_y
        )
        train_mse = model_obj.train_model()
        val_mse = model_obj.test_model(val_x, val_y)
        test_mse = model_obj.test_model(test_x, test_y)

        res_dict[columns[0]] = num_layers
        res_dict[columns[1]] = time_steps
        res_dict[columns[2]] = train_mse
        res_dict[columns[3]] = val_mse
        res_dict[columns[4]] = test_mse
        df = df.append(res_dict, ignore_index=True)

    df.to_csv('model_4_op.csv')
    return


experiment()