import numpy as np
from keras.models import load_model
from keras.layers import Dense
import h5py
import keras
import data_feeder_2 as data_feeder
from itertools import tee, izip
import keras.preprocessing.sequence
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


lstm_time_step = 8
batch_size = 32
epochs = 400
window_size = 8

def get_stacked_ae():

    if os.path.exists('ae_model.h5'):
        return

    nb_epoch = 500
    batch_size = 256
    X_train, _, _, _, _ = data_feeder.get_data(True)
    print X_train.shape

    num_layers = 2
    shape = [ 8, 4 ]
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

get_stacked_ae()

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
    for w in window(_arr, window_size + 1):
        inp.append(w[:-1])
        op.append(w[-1])

    inp = np.asarray(inp)
    op = np.reshape(np.asarray(op), [-1, 1])

    return inp, op


def create_complete_model():
    # ------------ #

    global lstm_time_step
    global batch_size
    global epochs
    global window_size

    # ------------ #

    if os.path.exists('model_1.h5'):
        return

    ae_model = load_model('ae_model.h5')
    X_train,_, Y_train, _, _ = data_feeder.get_data(True)


    # create windows
    inp, op = get_windowed_inp(Y_train, window_size)
    num_samples = inp.shape[0]
    # print 'num_samples', num_samples

    X_train = X_train[-num_samples:, :]
    # print 'X_train', X_train.shape
    # concatenate X_train and input passed through ae
    ae_op = ae_model.predict(X_train)
    # print 'ae_op', ae_op.shape

    inp = np.concatenate([inp, ae_op], axis=-1)
    # print 'inp shape', inp.shape

    # Reshape the data

    pad_len = lstm_time_step - (inp.shape[0] % lstm_time_step)

    # add padding
    _inp_pad = np.zeros(shape=[pad_len, inp.shape[-1]])
    _op_pad = np.zeros(shape=[pad_len, op.shape[-1]])

    inp = np.concatenate([_inp_pad, inp], axis=0)
    op = np.concatenate([_op_pad, op], axis=0)
    inp_dim = inp.shape[-1]
    op_dim = op.shape[-1]

    inp = np.reshape(inp, [-1, lstm_time_step, inp_dim])
    op = np.reshape(op, [-1, lstm_time_step, op_dim])


    # Ensure number of samples are divisible by batch size

    num_samples = (inp.shape[0] // batch_size)*batch_size
    inp = inp[-num_samples:,:,:]
    op = op[-num_samples:, :, :]


    model = keras.models.Sequential()


    batch_input_shape = [batch_size, inp.shape[1], inp.shape[-1]]
    lstm1 = keras.layers.LSTM(16,
                              use_bias=True,
                              batch_input_shape=batch_input_shape,
                              dropout=0.25,
                              stateful=True,
                              return_sequences=True,
                              )
    model.add(lstm1)
    lstm2 = keras.layers.LSTM(16,
                              use_bias=True,
                              dropout=0.25,
                              stateful=True,
                              return_sequences=True,
                              )
    model.add(lstm2)
    d1 = keras.layers.TimeDistributed(Dense(units=8, activation='tanh'))
    model.add(d1)
    d2 = keras.layers.TimeDistributed(Dense(units=1, activation='tanh'))
    model.add(d2)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MSE,
                  metrics=['mse'])

    # Ensure number of samples divisible by batch size

    history = model.fit(
        inp,
        op,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=False
    )

    t_loss = history.history['loss']
    plt.figure()
    plt.title('Training Loss', fontsize=20)
    plt.ylabel('Mean Square Error', fontsize=20)
    plt.plot(range(len(t_loss)), t_loss, 'r-')
    plt.xlabel('Epochs', fontsize=20)
    plt.yticks(np.arange(0, 2.2, 0.2))
    # plt.show()
    model.save('model_1.h5')
    return


    # plot loss !

def test_model():

    # -------- #

    global lstm_time_step
    global batch_size
    global epochs
    global window_size

    # --------- #

    ae_model = load_model('ae_model.h5')
    lstm_model = load_model('model_1.h5')

    # ---------- #

    _, X_test, _, Y_test, _ = data_feeder.get_data(True)

    # create windows
    inp, op = get_windowed_inp(Y_test, window_size)
    num_samples = inp.shape[0]
    print 'num_samples', num_samples

    X_test = X_test[-num_samples:, :]
    print 'X_test', X_test.shape
    # concatenate X_train and input passed through ae
    ae_op = ae_model.predict(X_test)
    # print 'ae_op', ae_op.shape

    inp = np.concatenate([inp, ae_op], axis=-1)
    print 'inp shape', inp.shape

    # Reshape the data

    pad_len = lstm_time_step - (inp.shape[0] % lstm_time_step)

    # add padding
    _inp_pad = np.zeros(shape=[pad_len, inp.shape[-1]])
    _op_pad = np.zeros(shape=[pad_len, op.shape[-1]])

    inp = np.concatenate([_inp_pad, inp], axis=0)
    op = np.concatenate([_op_pad, op], axis=0)
    inp_dim = inp.shape[-1]
    op_dim = op.shape[-1]

    inp = np.reshape(inp, [-1, lstm_time_step, inp_dim])
    op = np.reshape(op, [-1, lstm_time_step, op_dim])

    num_samples = (inp.shape[0] // batch_size) * batch_size
    inp = inp[-num_samples:, :, :]
    op = op[-num_samples:, :, :]

    # -------- #

    print 'Shape input dimension ', inp.shape
    print 'Shape output dimension ', op.shape

    test_x = inp
    test_y = op
    score = lstm_model.evaluate(x=test_x, y=test_y, batch_size=batch_size)
    print 'Mean Square Error', score[0]
    return score[0]


res_dict = {}
for w in [8,16,32,64,128]:
    # delete model
    import os
    if os.path.isfile('model_1.h5'):
        os.remove('model_1.h5')
    window_size = w

    get_stacked_ae()
    create_complete_model()
    mse = test_model()
    res_dict[w] = mse

print('Results ')
print( res_dict )