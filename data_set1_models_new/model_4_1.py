import keras.layers as layers
import keras.models as models
import keras
import numpy as np
import data_feeder
import utils
import math
import pandas as pd


# --------------------------- #
# CNN with both exogenous and endogenous data
# --------------------------- #

def get_data(time_window, type):
    # window of training data = time_window
    X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = data_feeder.get_data_val(std=True)


    if type == 'test':
        exg_x = X_test
        end_x = Y_test
    elif type == 'val':
        exg_x = X_val
        end_x = Y_val
    else:
        exg_x = X_train
        end_x = Y_train

    exg_x = utils.get_windowed_data_md(exg_x, time_window)

    # Reshape exg_shape to have same number of channels as exogenous dimensions
    dim = exg_x.shape[-1]
    exg_x = np.reshape(exg_x, [exg_x.shape[0], time_window, dim])
    # print 'exg_x shape', exg_x.shape
    y = np.asanyarray(end_x)
    y = np.reshape(y, [-1, 1])
    res = utils.get_windowed_data(y, time_window + 1)
    end_x = res[:, 0:time_window]
    y = res[:, -1:]
    end_x = np.reshape(end_x, [end_x.shape[0], time_window, 1])
    # print 'end_x shape', end_x.shape

    y = np.reshape(y, [y.shape[0], 1])

    num_samples = y.shape[0]
    exg_x = exg_x[0:num_samples]
    print exg_x.shape, end_x.shape, y.shape
    return exg_x, end_x, y

def get_val_data(time_window):
    type = "val"
    return get_data(time_window, type)

def get_test_data(time_window):
    type = "test"
    return get_data(time_window, type)


def get_training_data(time_window):
    type = "train"
    return get_data(time_window, type)


# --------------------------- #

class model:

    def __init__(self):
        self.model = None
        return

    def set_file_name(self):
        self.save_file = 'model_4' + str(self.time_window) + '.h5'

    def set_hyperparameters(self, batch_size, epochs, time_window, exog_dim, num_layers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.time_window = time_window
        self.num_layers = num_layers
        self.exog_dim = exog_dim
        self.set_file_name()
        return

    def set_train_data(self, train_x_end, train_x_exg, train_y):
        self.train_x_end = train_x_end
        self.train_x_exg = train_x_exg
        self.train_y = train_y

        return

    def block(self, inp, num_filters, dilation, kernel_size):
        # down-sampling is performed with a stride of 2
        conv_layer_op = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            activation = 'relu',
            strides=1,
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid'
        )(inp)

        res = layers.BatchNormalization()(conv_layer_op)
        res = layers.LeakyReLU()(res)
        return res

    def build(self):
        # Exogenous data Input data input
        input_layer_1 = layers.Input(
            shape=(self.time_window, self.exog_dim)
        )


        # input is an image of time_window x 1
        # Endogenous data Input
        input_layer_2 = layers.Input(
            shape=(self.time_window, 1)
        )



        exg_conv_layer_op_1 = layers.Conv1D(
            filters = 16,
            kernel_size=2,
            dilation_rate=[1],
            strides=1,
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid'
        )(input_layer_1)

        print exg_conv_layer_op_1
        exg_conv_layer_op_1 = layers.BatchNormalization()(exg_conv_layer_op_1)
        exg_conv_layer_op_1 = layers.LeakyReLU()(exg_conv_layer_op_1)

        end_conv_layer_op_1 = layers.Conv1D(
            filters=4,
            kernel_size=2,
            dilation_rate=[1],
            strides=1,
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid'
        )(input_layer_2)

        print end_conv_layer_op_1

        concat_op  = layers.Concatenate(axis=-1)([end_conv_layer_op_1,exg_conv_layer_op_1])
        print 'Added ', concat_op

        concat_op = layers.LeakyReLU()(concat_op)
        print concat_op
        # ---- #
        # inp = input_layer_1
        network_op = None

        inp = concat_op

        for l in range(1,self.num_layers):
            d = math.pow(2, l)
            network_op = self.block(
                inp=inp,
                num_filters=4,
                dilation=[d],
                kernel_size=2
            )
            print network_op
            inp = network_op

        network_op = layers.Conv1D(
            filters=1,
            kernel_size=1,
            dilation_rate=[1],
            strides=1,
            kernel_regularizer=keras.regularizers.l2(0.01),
            padding='valid')(inp)

        print 'OP', network_op
        network_op = layers.Flatten()(network_op)
        print 'OP' ,network_op


        model = models.Model(
            inputs=[input_layer_1,input_layer_2],
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
            'mse': 'mean_squared_error'
        }

        train_loss = []
        hist = self.model.fit(
            [self.train_x_exg, self.train_x_end],
            [self.train_y],
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=False
        )


        train_loss.extend(hist.history['loss'])
        print 'Train Loss', train_loss
        return np.mean(train_loss)



    def val_model(self, val_x_end, val_x_exg, val_y):

        score = self.model.evaluate(
            [val_x_end, val_x_exg],
            [val_y],
            batch_size=32
        )
        print 'Val Score', np.mean(score)
        return np.mean(score)





    def test_model(self, test_x_end, test_x_exg, test_y):

        score = self.model.evaluate(
            [test_x_end, test_x_exg],
            [test_y],
            batch_size=32
        )
        print 'Test Score', np.mean(score)
        return np.mean(score)


# ---------------------- #
def experiment():

    _num_layers = [3, 4, 6, 8, 10]

    batch_size = 64
    epochs = 150

    columns = [
        'Num Layers',
        'Time Window',
        'Train Error',
        'Validation Error',
        'Test Error'
    ]

    df = pd.DataFrame(columns=columns)

    for num_layers in _num_layers:

        time_steps = int(math.pow(2, num_layers))
        print 'time_steps ', time_steps

        train_x_exg, train_x_end, train_y = get_training_data(time_steps)
        val_x_end, val_x_exg, val_y = get_val_data(time_steps)
        test_x_exg, test_x_end, test_y = get_test_data(time_steps)

        exog_dim = 81
        model_obj = model()
        model_obj.set_hyperparameters(
            epochs=epochs,
            batch_size=batch_size,
            time_window=time_steps,
            exog_dim=exog_dim,
            num_layers=num_layers
        )
        model_obj.build()

        model_obj.set_train_data(
            train_x_end, train_x_exg, train_y
        )
        train_mse = model_obj.train_model()
        val_mse = model_obj.val_model( val_x_end, val_x_exg, val_y)
        test_mse = model_obj.test_model( test_x_exg, test_x_end, test_y)

        res_dict = {}
        res_dict[columns[0]] = num_layers
        res_dict[columns[1]] = time_steps
        res_dict[columns[2]] = train_mse
        res_dict[columns[3]] = val_mse
        res_dict[columns[4]] = test_mse
        df = df.append(res_dict, ignore_index=True)
        print res_dict


    df.to_csv('model_4_1_op.csv')
    return


experiment()
