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

import tensorflow as tf
import numpy as np
import sklearn
from sklearn.svm import SVR
# design the autoencoder #

class ae:

    def __init__(self):
        self.set_hyperparams()
        return

    def layerwise_train(self):
        for i in range(self.num_hidden_layers):
            print 'Training hidden layer ',i+1
            self.build_till_layer(i)
            self.session()
            self.train_model()

    def set_data(self,x):
        self.data_x = x

    def session(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def set_hyperparams(self):
        self.inp_dim = 81
        self.batch_size = 256
        self.layer_dims = [64,32]
        self.num_hidden_layers = len(self.layer_dims)
        self.weights = [None] * self.num_hidden_layers
        self.bias_enc = [None] * self.num_hidden_layers
        self.bias_dec = [None] * self.num_hidden_layers
        self.is_trained = [False] * self.num_hidden_layers


    def build_till_layer(self, till_layer ):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None,self.inp_dim])

        inp = self.x

        if till_layer == 0:
            self.weights[0] = tf.Variable(initial_value = tf.random_normal([self.inp_dim, self.layer_dims[0]], stddev=0.25),trainable=True)
            self.bias_enc[0] = tf.Variable( initial_value=tf.random_normal([self.layer_dims[0]],stddev=0.25), trainable=True)
            op1 = tf.nn.xw_plus_b(inp,self.weights[0],self.bias_enc[0])
            self.bias_dec[0] = tf.Variable(initial_value=tf.random_normal([self.inp_dim], stddev=0.25),trainable=True)
            op2 = tf.nn.xw_plus_b(op1,tf.transpose(self.weights[0],[1,0]),self.bias_dec[0])

            self.y1 = op2
            self.y2 = inp
            self.add_train_to_graph()
            return

        if till_layer == 1:
            # freeze the weights
            self.weights[0] = tf.Variable(initial_value = self.weights[0] ,trainable=False)
            self.bias_enc[0] = tf.Variable(initial_value= self.bias_enc[0],trainable=False)

            op1 = tf.nn.xw_plus_b(inp, self.weights[0], self.bias_enc[0])
            self.op = op1
            self.weights[1] = tf.Variable(initial_value=tf.random_normal([self.layer_dims[0], self.layer_dims[1]], stddev=0.25),
                                          trainable=True)
            self.bias_enc[1] = tf.Variable(initial_value=tf.random_normal([self.layer_dims[1]], stddev=1),
                                           trainable=True)
            self.bias_dec[1] = tf.Variable(initial_value=tf.random_normal([self.layer_dims[0]], stddev=1), trainable=True)

            op2 = tf.nn.xw_plus_b(op1,self.weights[1],self.bias_enc[1])
            op3 = tf.nn.xw_plus_b(op2,tf.transpose(self.weights[1],[1,0]),self.bias_dec[1])

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
        num_batches = ((data_len-1)//self.batch_size)+1
        b_data = []
        for i in range(num_batches):
            b_data.append(self.data_x[i*self.batch_size:(i+1)*self.batch_size])
        return b_data

    def train_model(self):
        print self.weights
        batched_data = self.create_batches()
        for d in batched_data:
            l ,_ = self.sess.run([self.loss, self.t] , feed_dict={self.x : d})
            print 'Loss in training ',l

    def reduce(self,data):
        output = self.sess.run(self.op, feed_dict={self.x: data})
        return output


data_x = np.random.randn(1000,81)
ae_obj = ae()
ae_obj.set_data(data_x)
ae_obj.layerwise_train()
ae_op = ae_obj.reduce(data_x)


#-----------------#
data_y1 =  np.random.randn(1000,30)
data_y2 = np.random.randn(1000,1)
model = SVR(kernel='rbf',
            C=1.0,
            gamma = 1.0,
            epsilon=0.2,
            verbose = True
            )
model.fit(X=data_y1, y=data_y2)
model.predict(data_y1)