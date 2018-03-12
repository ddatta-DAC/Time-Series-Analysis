import tensorflow as tf
import numpy as np
import data_feeder

class hc_LSTMCell (tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None):
        self._num_units = num_units
        super (hc_LSTMCell, self).__init__ (num_units, forget_bias,
                                            state_is_tuple, activation, reuse)

    #     @property
    #     def state_size(self):
    #         return 2 * self._num_units

    @property
    # Return h + c
    def output_size(self):
        return 2 * self._num_units

    def call(self, inputs, inp_state):
        with tf.variable_scope (type (self).__name__):
            op, cell_state = super (hc_LSTMCell, self).call (inputs, inp_state)
            h_c = tf.concat ([inp_state.h, inp_state.c], axis=1)
            return h_c, cell_state


class model:
    def __init__(self):
        self.name = 'Model'
        self.scope = 'Scope_1'

        # tf.reset_default_graph()
        # self.graph = tf.Graph()
        # self.session = tf.Session(graph=self.graph)

        with tf.variable_scope(self.scope):
            self.set_hyper_params()
            self.build()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        return



    def set_hyper_params(self):
        self.rnn_timesteps = 8
        self.time_window = 10
        self.num_exog = 81
        self.dropout_prob = 0.15
        self.rnn_1_size = 64
        return

    def close_session(self):
        writer = tf.summary.FileWriter('./graphs', self.session.graph)
        writer.close()
        self.session.close()
        # use : tensorboard --logdir="./graphs" --port 6006
        return


    def get_lstm_layer(self, num_layers, rnn_size, state_is_tuple=True):
        cells = []
        for i in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell (
                rnn_size,
                forget_bias=1.0,
                state_is_tuple=state_is_tuple
            )
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob)
            cells.append(cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
        return rnn_cell


    def build(self):
        self.x_exog = tf.placeholder(dtype=tf.float32, shape=[None,self.time_window,self.num_exog])

        self.rnn_1 = self.get_lstm_layer( num_layers = 1, rnn_size = self.rnn_1_size , state_is_tuple = False )
        rnn_1_input = self.x_exog
        self.rnn_1_last_state = None


        if self.rnn_1_last_state is None:
            init_state = None
        else:
            init_state = tuple(tf.unstack(self.rnn_1_last_state, axis=1))

        with tf.variable_scope ('rnn_1'):
            self.rnn_1_op, self.rnn_1_state = tf.nn.dynamic_rnn (
                cell = self.rnn_1,
                inputs = rnn_1_input,
                initial_state = init_state,
                dtype = tf.float32
            )


        # print rnn_1_input.shape
        print 'rnn_1_op : ', self.rnn_1_op.shape
        # print 'rnn_1_state : ', len(self.rnn_1_state) , self.rnn_1_state[0].c.shape, self.rnn_1_state[0].h.shape
        print 'rnn_1_state : ', self.rnn_1_state.shape

        # First attention layer

        # Need to initialize with 0s
        self.rnn_1_prev_op = tf.placeholder(dtype=tf.float32, shape=[None, self.time_window, 2*self.rnn_1_size])

        W_e_dim = [self.time_window , 2 * self.rnn_1_size ]
        b_e_dim = [self.time_window]
        U_e_dim = [self.time_window,self.time_window]
        v_e_dim = [self.time_window,1]
        W_e = tf.Variable(dtype=tf.float32,initial_value=tf.random_normal (W_e_dim, stddev=0.1), trainable=True)
        b_e = tf.Variable(dtype=tf.float32,initial_value=tf.random_normal (b_e_dim, stddev=0.1), trainable=True)
        U_e = tf.Variable(dtype=tf.float32,initial_value=tf.random_normal (U_e_dim, stddev=0.1), trainable=True)
        v_e = tf.Variable(dtype=tf.float32,initial_value=tf.random_normal (v_e_dim, stddev=0.1), trainable=True)
        # W_e *[h(t);c(t)] + U_e * x(1...t)



        z1 = tf.matmul(a=self.rnn_1_state,b=W_e,transpose_b=True)
        print 'z1 shape', z1.shape
        # unstack along each input series dim
        unstacked_x = tf.unstack(tf.transpose(rnn_1_input,[0,2,1]),axis=1)
        list_tmp = []
        for _x_k in unstacked_x:
            tmp = tf.matmul(a=_x_k,b=U_e)
            tmp = tmp + z1 + b_e
            list_tmp.append(tmp)
        x_k = tf.stack(list_tmp,axis=2)
        z2 = tf.tanh(x_k)
        z2 = tf.unstack(tf.transpose(z2,[0,2,1]),axis=1)
        list_z2 = []
        for _z2 in z2 :
            _z2 = tf.matmul(_z2,v_e)
            list_z2.append(_z2)

        z2 = tf.stack(list_z2, axis = 1 )
        print z2.shape
        att_wts = tf.squeeze(tf.nn.softmax(z2),axis=2)
        print 'Shape of attention weights' ,att_wts.shape

        encoder_op = (tf.transpose(rnn_1_input,[0,2,1]))
        encoder_op = encoder_op [:,:,-1]
        print encoder_op.shape
        encoder_op = tf.multiply(encoder_op, att_wts)
        print ' Shape of encoder outputs', encoder_op.shape




model()






