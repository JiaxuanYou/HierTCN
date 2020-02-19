import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *

# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20


def model_gru(args, x, x_gap=None,x_impression=None, initial_state=None,return_state=False, name='gru',reuse=tf.AUTO_REUSE):
    """
    given a batch of sequence x, pred a batch of sequence y_hat using GRU
    :param x: (b,t,d)
    :param args: object that contains all setup parameters (import from args.py)
    :param x_gap: (b,t,1)
    :param x_impression: (b,t,k)
    :param initial_state: iniital hidden state: (b,num_layer*num_hidden_dim)
    :param return_state: whether return the hidden state
    :param name: default name
    :param reuse: defualt auto_reuse
    :return: pred: (b,t,d)
    """
    with tf.variable_scope(name,reuse=reuse):

        if args.has_gap and x_gap is not None:
            x_weight = tf.exp(-x_gap / args.gap_bandwidth)
            x = tf.concat([x, x_weight],axis=2)
        if args.has_impression and x_impression is not None:
            x = tf.concat([x, tf.reduce_mean(x_impression,axis=2)],axis=2)

        rnn_layers = []
        for i in range(args.num_layer):
            if args.has_batchnorm:
                rnn_layers.append(GRUCellBN(args.hidden_dim,has_weightnorm=args.has_weightnorm))
            else:
                rnn_layers.append(rnn.GRUCell(args.hidden_dim))
        if args.has_batchnorm:
            multi_rnn_cell = MultiRNNCellBN(rnn_layers, state_is_tuple=False)
        else:
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers, state_is_tuple=False)

        x = dense(x, units=128, activation=None, use_bias=False, has_weightnorm=args.has_weightnorm, reuse=tf.AUTO_REUSE, name='emb')
        pred, state = tf.nn.dynamic_rnn(
            multi_rnn_cell,
            x,
            dtype=tf.float32,
            sequence_length=length(x),
            initial_state=initial_state
        )
        # pred = tf.contrib.layers.fully_connected(pred,num_outputs=args.output_dim,activation_fn=None)
        pred = dense(pred,units=args.output_dim,activation=None,has_weightnorm=args.has_weightnorm)
        if args.l2_normalize:
            pred = tf.nn.l2_normalize(pred,dim=-1)
        if return_state:
            return pred,state
        else:
            return pred


def model_gru_gap_weight(args, x, x_gap=None, x_impression=None, name='gru_weight',reuse=tf.AUTO_REUSE):
    """
    gru model with time gap between the transition of hidden states
    :param x: (b,t,d)
    :param args: object that contains all setup parameters (import from args.py)
    :param x_weight:(b,t,1)
    :return: pred : (b,t,d)
    """
    with tf.variable_scope(name, reuse=reuse):

        if args.has_gap and x_gap is not None:
            x_weight = tf.exp(-x_gap / args.gap_bandwidth)
        if args.has_impression and x_impression is not None:
            x = tf.concat([x, tf.reduce_mean(x_impression,axis=2)],axis=2)

        # rnn cell
        rnn_layers = []
        for i in range(args.num_layer):
            rnn_layers.append(rnn.GRUCell(args.hidden_dim))
        multi_rnn_cell = rnn.MultiRNNCell(rnn_layers, state_is_tuple=False)

        # initial state
        init_state = multi_rnn_cell.zero_state(args.batch_size, dtype=tf.float32)

        # loop over time steps
        pred = []
        for t in range(args.max_seq_len-1): #
            output, state= multi_rnn_cell(x[:,t,:], init_state) # x[:,t,:] is b*d, state is b*(hidden_dim*layer)
            if t != args.max_seq_len-2: # the last idx
                init_state = state*x_weight[:,t+1,:] # x_weight[:,t,:] is b*1, init_state is b*(hidden_dim*layer)
            pred.append(output)

        # Transform the output to b*t*d
        pred = tf.transpose(tf.squeeze(tf.stack(pred)), [1, 0, 2])

        # pred = tf.contrib.layers.fully_connected(pred,num_outputs=args.output_dim,activation_fn=None)
        pred = dense(pred, units=args.output_dim, activation=None, has_weightnorm=args.has_weightnorm)
        if args.l2_normalize:
            pred = tf.nn.l2_normalize(pred,dim=-1)
        return pred



def length(sequence):
    """
    compute actual length of sequence (non-zero timesteps) in each batch
    :param sequence: (b,t,d)
    :return: length: (b,)
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2)) # b*t
    length = tf.reduce_sum(used, 1)# b
    length = tf.cast(length, tf.int32)
    return length
