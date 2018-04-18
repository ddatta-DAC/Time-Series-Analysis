from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn

# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
# import tensorflow.rnn.python.ops.rnn_cell as rnn_cell

import attention_encoder
import Generate_stock_data as GD

# Parameters
learning_rate = 0.01
training_iters = 500
batch_size = 128
display_step = 100
model_path = "./stock_dual/"

# Network Parameters
# encoder parameter
n_input_encoder = 81  # n_feature of encoder input
n_steps_encoder = 128  # time steps
n_hidden_encoder = 128  # size of hidden units

# decoder parameter
n_input_decoder = 1
n_steps_decoder = 127
n_hidden_decoder = 128
n_classes = 1  # size of the decoder output

# tf Graph input
encoder_input = tf.placeholder("float", [None, n_steps_encoder, n_input_encoder])
decoder_input = tf.placeholder("float", [None, n_steps_decoder, n_input_decoder])
decoder_gt = tf.placeholder("float", [None, n_classes])
encoder_attention_states = tf.placeholder("float", [None, n_input_encoder, n_steps_encoder])

# Define weights
weights = {'out1': tf.Variable(tf.random_normal([n_hidden_decoder, n_classes]))}
biases = {'out1': tf.Variable(tf.random_normal([n_classes]))}


def RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Prepare data for encoder
    # Permuting batch_size and n_steps
    encoder_input = tf.transpose(encoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    encoder_input = tf.reshape(encoder_input, [-1, n_input_encoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    encoder_input = tf.split(encoder_input, n_steps_encoder, 0)

    # Prepare data for decoder
    # Permuting batch_size and n_steps
    decoder_input = tf.transpose(decoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    decoder_input = tf.reshape(decoder_input, [-1, n_input_decoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    decoder_input = tf.split(decoder_input, n_steps_decoder, 0)

    # Encoder.
    with tf.variable_scope('encoder') as scope:
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
        encoder_outputs, encoder_state, attn_weights = attention_encoder.attention_encoder(encoder_input,
                                                                                           encoder_attention_states,
                                                                                           encoder_cell)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
    attention_states = tf.concat(top_states, 1)

    with tf.variable_scope('decoder') as scope:
        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
        outputs, states = seq2seq.attention_decoder(decoder_input, encoder_state,
                                                    attention_states, decoder_cell)

    return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights


pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states)
# Define loss and optimizer
cost = tf.reduce_sum(tf.pow(tf.subtract(pred, decoder_gt), 2))
loss = tf.pow(tf.subtract(pred, decoder_gt), 2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

# save the model
saver = tf.train.Saver()
loss_value = []
step_value = []
loss_test = []
loss_val = []
test_MSE = 0


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    count = 1

    # read the input data
    Data = GD.Input_data(batch_size, n_steps_encoder, n_steps_decoder, n_hidden_encoder)
    # Keep training until reach max iterations
    train_loss = []
    while step < training_iters:
        # the shape of batch_x is (batch_size, n_steps, n_input)
        batch_x, batch_y, prev_y, encoder_states = Data.next_batch()
        feed_dict = {encoder_input: batch_x,
                     decoder_gt: batch_y,
                     decoder_input: prev_y,
                     encoder_attention_states: encoder_states
                     }
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict)

        _loss =  sess.run(cost, feed_dict) / batch_size
        train_loss.append(_loss)
        # display the result
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(cost, feed_dict) / batch_size
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss))

            # store the value
            loss_value.append(loss)
            step_value.append(step)
            # Val
            val_x, val_y, val_prev_y, encoder_states_val = Data.validation()
            feed_dict = {encoder_input: val_x, decoder_gt: val_y, decoder_input: val_prev_y,
                         encoder_attention_states: encoder_states_val}
            loss_val1 = sess.run(cost, feed_dict) / len(val_y)
            loss_val.append(loss_val1)
            print("validation MSE:", loss_val1)

            # testing
            test_x, test_y, test_prev_y, encoder_states_test = Data.testing()
            feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                         encoder_attention_states: encoder_states_test}
            pred_y = sess.run(pred, feed_dict)
            loss_test1 = sess.run(cost, feed_dict) / len(test_y)
            loss_test.append(loss_test1)
            print("Testing MSE:", loss_test1)
            test_MSE = loss_test1

            # save the parameters
            if loss_val1 <= min(loss_val):
                save_path = saver.save(sess, model_path + 'dual_stage_' + str(step) + '.ckpt')

        step += 1
        count += 1

        # reduce the learning rate
        if count > 10000:
            learning_rate *= 0.1
            count = 0
            save_path = saver.save(sess, model_path + 'dual_stage_' + str(step) + '.ckpt')

    print ("Optimization Finished!")
    print (train_loss)

    import utils

    utils.plot2(range(len(train_loss)), train_loss, 'Iteration', 'MSE', 'Training Error for Model 8')
    print ('MSE test', test_MSE)
