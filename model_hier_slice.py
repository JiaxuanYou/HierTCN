import tensorflow as tf
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *

from model_gru import *
from model_gru_gmm import *
from model_tcn import *
from model_tcn_gmm import *
from model_mv import *
from utils import *


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# D: topic dimension, eg, 50
# k: number of impressions, eg 20
# m: number of sessions, eg 10
# s: length of session, eg 100


def model_hier_slice(args,x,y,mask,state,y_slice,topic_state,x_gap=None,x_impression=None,name='hier_slice',reuse=tf.AUTO_REUSE,training=tf.constant(True)):
    """
    :param x: m*(b,s,d)
    :param y: m*(b,s,d)
    :param mask: m*(b,1)
    :param state: (b,num_hidden_dim*num_layer)
    :param y_slice: (b,d)
    :param topic_state: (b,D)
    :return: pred,state,y_slice,topic_state,topic_slice
    """
    with tf.variable_scope(name,reuse=reuse):
        # init high level rnn
        rnn_layers = []
        for i in range(args.num_layer):
            if args.has_batchnorm:
                rnn_layers.append(GRUCellBN(args.hidden_dim,has_weightnorm=args.has_weightnorm))
                multi_rnn_cell = MultiRNNCellBN(rnn_layers, state_is_tuple=False)
            else:
                rnn_layers.append(rnn.GRUCell(args.hidden_dim))
                multi_rnn_cell = rnn.MultiRNNCell(rnn_layers,state_is_tuple=False)

        # collect topic_state
        topic_state_list = [topic_state]

        # loop over user sessions
        for i in range(len(x)):
            mask_y = tf.sign(tf.reduce_sum(tf.abs(y[i]), 2))  # B*T
            # low level model input
            x_slice = x[i]
            if args.data_noise is not None:
                x_slice += tf.random_normal(shape=tf.shape(x_slice), mean=0.0, stddev=args.data_noise, dtype=tf.float32)
            # state
            state_slice = tf.tile(tf.expand_dims(state,axis=1),[1,tf.shape(x_slice)[1],1])
            x_slice = tf.concat((x_slice,state_slice),axis=-1)
            # y_slice
            y_slice = tf.tile(tf.expand_dims(y_slice, axis=1), [1, tf.shape(x_slice)[1], 1])
            x_slice = tf.concat((x_slice, y_slice), axis=-1)
            # topic_slice
            topic_state_tile = tf.tile(tf.expand_dims(topic_state, axis=1), [1, tf.shape(x_slice)[1], 1])
            x_slice = tf.concat((x_slice, topic_state_tile), axis=-1)
            x_slice = dense(x_slice, units=256, activation=None,
                            has_weightnorm=args.has_weightnorm, reuse=tf.AUTO_REUSE, name='x_slice')
            if args.has_impression:
                impression_slice = tf.reduce_mean(x_impression[i],axis=2)
                x_slice = tf.concat((x_slice, impression_slice), axis=2)
            # low level model
            if args.model_low_type =='tcn':
                pred = model_tcn(args,x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'gru':
                pred = model_gru(args,x_slice)
            elif args.model_low_type=='tcn_gmm':
                pred = model_tcn_gmm(args,x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'gru_gmm':
                pred = model_gru_gmm(args,x_slice)
            elif args.model_low_type == 'mv':
                x_pad = dense(state,units=args.output_dim,activation=None,has_weightnorm=args.has_weightnorm)
                pred = model_mv_tf(x[i],x_pad,n=args.lag)
            # save pred for each session
            if i==0:
                pred_all = pred
            else:
                pred_all = tf.concat((pred_all,pred),axis=1)
            # high level model input
            # y_slice = tf.reduce_max(y[i], axis=1)  # B*512
            # mean pool
            y_activity_count = tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(y[i]), 2)),1,keep_dims=True)
            y_slice = tf.reduce_sum(y[i], axis=1)/y_activity_count  # B*512
            # high level model update
            if args.has_batchnorm:
                output, state = multi_rnn_cell(y_slice, state, training=training)
            else:
                output, state = multi_rnn_cell(y_slice, state)

            # get the topic_state
            topic_state = topic_update(y_slice, topic_state)

            if args.has_gap:
                # apply decay
                state *= tf.exp(-x_gap[i] / args.gap_bandwidth)

            # apply mask
            state *= mask[i]
            y_slice *= mask[i]
            topic_state *= mask[i]
            # if a row in topic_state is 0, it means the user is changed, and we make topic_state uniform
            topic_mask = (1 - mask[i]) * 1.0/args.topic_dim
            topic_state += topic_mask

            topic_state_list.append(topic_state)

        return pred_all,state,y_slice,topic_state,topic_state_list


def topic_update(y_slice, topic_state, name='topic', reuse=tf.AUTO_REUSE):
    """
    first update the topic_state: pi_t = (1-lambda)*f(y_t)+lambda*pi_{t-1}
    then compute topic_slice: y'_t = g(pi_t)
    :param y_slice: b*10
    :param topic_state: b*512
    :return: topic_state: b*10
    """
    with tf.variable_scope(name,reuse=reuse):
        # out = tf.contrib.layers.fully_connected(y_slice, num_outputs=args.topic_dim+1, activation_fn=None)  # b,D+1
        # current_topic_state, topic_lambda = tf.split(out, num_or_size_splits=[args.topic_dim, 1], axis=-1)
        current_topic_state = tf.contrib.layers.fully_connected(y_slice, num_outputs=args.topic_dim, activation_fn=None)  # b,D
        current_topic_state = tf.nn.softmax(current_topic_state)  # b,D
        topic_lambda = tf.sigmoid(tf.get_variable(name='topic_lambda',shape=[])) # shared among users
        topic_state = (1 - topic_lambda) * current_topic_state + topic_lambda * topic_state # b,D
    return topic_state