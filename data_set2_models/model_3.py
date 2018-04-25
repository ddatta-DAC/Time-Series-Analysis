'''

 model description :

 There are exogenous inputs X [x1,x2...xk]
 Train a stacked auto encoder on these.
 z
 Then use a SVR (Support vector regression, with a window size : w
  Dependent variables :
  - [ y(t-w1), y(t-w1+1), ... y(t-1)]
  - [ z[t-w2), z(t-w2+1), ... z(t-1)]

'''

import data_feeder_2 as data_feeder
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.svm import SVR
from itertools import tee, izip
import matplotlib.pyplot as plt
import os

# -- Common hyperparams -- #

window_size = 32
ae_max_epochs = 10

# ------------------------------------------------------------ #
# design the autoencoder #

class ae:

    def __init__(self):
        self.set_hyperparams()
        self.max_epochs = ae_max_epochs
        return

    def layerwise_train(self):
        for i in range(self.num_hidden_layers):
            print 'Training hidden layer ', i + 1
            self.build_till_layer(i)
            self.session()
            self.train_model()

    def set_data(self, x):
        self.data_x = x

    def session(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def set_hyperparams(self):
        self.inp_dim = 5
        self.batch_size = 64
        self.layer_dims = [8, 4]
        self.num_hidden_layers = len(self.layer_dims)
        self.weights = [None] * self.num_hidden_layers
        self.bias_enc = [None] * self.num_hidden_layers
        self.bias_dec = [None] * self.num_hidden_layers
        self.is_trained = [False] * self.num_hidden_layers

    def build_till_layer(self, till_layer):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.inp_dim])
        inp = self.x

        if till_layer == 0:
            self.weights[0] = tf.Variable(
                initial_value=tf.random_normal([self.inp_dim, self.layer_dims[0]], stddev=0.25), trainable=True)
            self.bias_enc[0] = tf.Variable(initial_value=tf.random_normal([self.layer_dims[0]], stddev=0.25),
                                           trainable=True)
            self.bias_dec[0] = tf.Variable(initial_value=tf.random_normal([self.inp_dim], stddev=0.25), trainable=True)

            op1 = tf.tanh(tf.nn.xw_plus_b(inp, self.weights[0], self.bias_enc[0]))
            op2 = tf.tanh(tf.nn.xw_plus_b(op1, tf.transpose(self.weights[0], [1, 0]), self.bias_dec[0]))

            self.y1 = op2
            self.y2 = inp
            self.add_train_to_graph()
            return

        if till_layer == 1:
            # freeze the weights
            self.weights[0] = tf.Variable(initial_value=self.weights[0], trainable=False)
            self.bias_enc[0] = tf.Variable(initial_value=self.bias_enc[0], trainable=False)

            op1 = tf.tanh(tf.nn.xw_plus_b(inp, self.weights[0], self.bias_enc[0]))
            self.op = op1

            self.weights[1] = tf.Variable(
                initial_value=tf.random_normal([self.layer_dims[0], self.layer_dims[1]], stddev=0.25),
                trainable=True)
            self.bias_enc[1] = tf.Variable(
                initial_value=tf.random_normal([self.layer_dims[1]], stddev=1),
                trainable=True)
            self.bias_dec[1] = tf.Variable(
                initial_value=tf.random_normal([self.layer_dims[0]], stddev=1),
                trainable=True)

            op2 = tf.tanh(tf.nn.xw_plus_b(op1, self.weights[1], self.bias_enc[1]))
            op3 = tf.tanh(tf.nn.xw_plus_b(op2, tf.transpose(self.weights[1], [1, 0]), self.bias_dec[1]))

            self.y1 = op1
            self.y2 = op3
            # output of dimensionality reduction
            self.op = op2

            self.add_train_to_graph()
            return

    def add_train_to_graph(self):
        self.loss = tf.losses.mean_squared_error(labels=self.y1, predictions=self.y2)
        self.opt = tf.train.AdamOptimizer()
        self.t = self.opt.minimize(self.loss)

    def create_batches(self):
        data_len = self.data_x.shape[0]
        num_batches = ((data_len - 1) // self.batch_size) + 1
        b_data = []
        for i in range(num_batches):
            b_data.append(self.data_x[i * self.batch_size:(i + 1) * self.batch_size])
        return b_data

    def train_model(self):
        print self.weights
        for ep in range(self.max_epochs):
            batched_data = self.create_batches()
            for d in batched_data:
                l, _ = self.sess.run([self.loss, self.t], feed_dict={self.x: d})
                print 'Loss in training ', l

    def reduce(self, data):
        output = self.sess.run(self.op, feed_dict={self.x: data})
        return output


def pretrain_ae(X_train):
    data_x = X_train
    ae_obj = ae()
    ae_obj.set_data(data_x)
    ae_obj.layerwise_train()
    ae_op = ae_obj.reduce(X_train)
    print ae_op.shape
    return ae_obj, ae_op

# --------------------------------------------------------------------------------------- #




# ----------------- #
# Create sliding window of target value #
# ----------------- #


def get_windowed_y( data , window_size):
    y = list(np.reshape(data, [data.shape[0]]))
    print len(y)

    # local function
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)

    inp = []
    op = []
    for w in window(y, window_size+1):
        inp.append(w[:-1])
        op.append(w[-1])

    inp = np.asarray(inp)
    op = np.reshape(np.asarray(op),[-1,1])
    print inp.shape
    print op.shape
    return inp, op


def get_train_data():
    X_train, X_test, Y_train, Y_test, _ = data_feeder.get_data(True)
    # set up train data
    ae_obj, X_train_ae = pretrain_ae(X_train)
    train_y_inp, train_y_op = get_windowed_y (Y_train, window_size = window_size)
    train_exog_inp = X_train_ae[window_size:,:]
    print ' train_exog_inp ', train_exog_inp.shape
    # concatenate train_exog_inp and train_y_op
    train_data_x = np.concatenate([train_y_inp,train_exog_inp],axis = 1)
    print 'train_data_x', train_data_x.shape
    print 'Train Data'
    print 'Train_data_x ', train_data_x.shape
    print 'train_data_x', train_y_op.shape

    # set up test data the model

    test_y_inp,test_y_op = get_windowed_y(Y_test,window_size = window_size)
    X_test_ae = ae_obj.reduce(X_test)
    test_exog_inp = X_test_ae[window_size:,:]
    # concatenate train_exog_inp and train_y_op
    test_data_x = np.concatenate([test_y_inp,test_exog_inp],axis = 1)
    print 'Test Data'
    print test_data_x.shape
    print test_y_op.shape

def get_svr(type = 'poly'):

    if type =='poly':
        model = SVR(kernel='poly',
                    C=1.0,
                    degree=5,
                    epsilon=0.2,
                    verbose=True
                    )
        return model
    if type =='rbf':
        model = SVR(kernel='rbf',
                    C=1.0,
                    gamma=1.0,
                    epsilon=0.2,
                    verbose=True
                    )
        return model


def experiment():
    global window_size

    _window_size = [ 32, 64, 128]
    k_type = ['poly', 'rbf']
    for k in k_type:


        for w in _window_size:
            window_size = w
            model = get_svr(k)

            # Train
            history = model.fit( X = train_data_x, y = train_y_op)
            train_loss = history.history['loss']
            print 'Test Mean Square Error' , train_loss
            # Test
            pred_res = model.predict(test_data_x)
            test_mse = sklearn.metrics.mean_squared_error(pred_res,test_y_op)
            print 'Testing Mean Square Error ', test_mse






# y_scaler_obj = scaler_array[-1]


# unscaled_pred_result = [ z for z in y_scaler_obj.inverse_transform(pred_res)]
# unscaled_y = [ z for z in y_scaler_obj.inverse_transform(test_y_op) ]
# mse = sklearn.metrics.mean_squared_error(pred_res , test_y_op)
# print mse



# # plot the graph!
# disp_x = range(len(unscaled_y))
# plt.plot(disp_x,unscaled_y,'r-')
# plt.plot(disp_x,unscaled_pred_result,'b-')
# plt.show()





# data_y1 = np.random.randn(1000, 30)
# data_y2 = np.random.randn(1000, 1)
# model = SVR(kernel='rbf',
#             C=1.0,
#             gamma=1.0,
#             epsilon=0.2,
#             verbose=True
#             )
#
# model.fit(X=data_y1, y=data_y2)
# model.predict(data_y1)
