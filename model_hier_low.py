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


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# D: topic dimension, eg, 50
# k: number of impressions, eg 20
# m: number of sessions, eg 10
# s: length of session, eg 100



def model_hier_low(args,x,name='hier_low',reuse=tf.AUTO_REUSE):
    """
    :param x: m*(b,s,d)
    :return: pred
    """
    with tf.variable_scope(name,reuse=reuse):
        # loop over user sessions
        for i in range(len(x)):
            # low level model input
            x_slice = x[i]
            # low level model
            if args.model_low_type=='tcn':
                pred = model_tcn(x_slice)
            elif args.model_low_type == 'gru':
                pred = model_gru(x_slice)
            elif args.model_low_type == 'mv':
                # x_pad = tf.contrib.layers.fully_connected(state,num_outputs=args.output_dim,activation_fn=None)
                x_pad = tf.get_variable('x_pad',shape=(args.batch_size, args.hidden_dim * args.num_layer))
                pred = model_mv_tf(x[i],n=args.lag,x_pad=x_pad)
            # save pred for each session
            if i==0:
                pred_all = pred
            else:
                pred_all = tf.concat((pred_all,pred),axis=1)
        return pred_all

