import tensorflow as tf
import numpy as np
from args import *
from utils import *
from scipy import stats

# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20


# ## tensorflow version of moving average
# def model_mv_tf_nopad(x,n,band_width=3.0):
#     """
#     for each batch, do the matrix level moving average
#     :param x: (b,t,d)
#     :param n: lags (previous timesteps) in moving average
#     :param band_width: the w in exp(-t/w)
#     :return: pred (b,t,d)
#     """
#     x_out_list = []
#     weigh_sum = sum([np.exp((-n+i)/(n/band_width)) for i in range(n)])
#     for i in range(n):
#         # x_out[:,i:-n+i,i,:] = x
#         pad = np.zeros((3,2))
#         pad[1,0]=n-i-1
#         pad[1,1]=i
#         x_pad = tf.pad(x,tf.constant(pad,dtype=tf.int32))
#         x_pad *= (np.exp((-n+i)/(n/band_width)))/weigh_sum
#         # print((np.exp((-n+i)/(n/band_width)))/weigh_sum)
#         x_out_list.append(x_pad)
#     x_out = tf.reduce_sum(tf.stack(x_out_list,axis=2),axis=2)
#     return x_out[:,:-n+1,:]

## tensorflow version of moving average
def model_mv_tf_nopad(x,n):
    """
    for each batch, do the matrix level moving average
    :param x: (b,t,d)
    :param n: lags (previous timesteps) in moving average
    :param band_width: the w in exp(-t/w)
    :return: pred (b,t,d)
    """
    x_out_list = []
    for i in range(n):
        pad = np.zeros((3,2))
        pad[1,0]=n-i-1
        pad[1,1]=i
        x_pad = tf.pad(x,tf.constant(pad,dtype=tf.int32))
        x_out_list.append(x_pad)
    x_out = tf.reduce_mean(tf.stack(x_out_list,axis=2),axis=2)
    return x_out[:,:-n+1,:]



def model_mv_tf(x,n,x_pad=None):
    """
    for each batch, do the matrix level moving average
    :param x: (b,t,d)
    :param x_pad (b,d)
    :param n: lags (previous timesteps) in moving average
    :param band_width: the w in exp(-t/w)
    :return: pred (b,t,d)
    """
    # if x_pad is None:
    x_pad = tf.zeros((tf.shape(x)[0],n,tf.shape(x)[-1]))
    # x_pad = tf.random_uniform((tf.shape(x)[0],n,tf.shape(x)[-1]))
    # x_pad = x_pad / tf.reduce_sum(x_pad,axis=-1,keep_dims=True)
    x = tf.concat((x_pad,x),axis=1)
    x_out = model_mv_tf_nopad(x,n)
    return x_out[:,n:,:]

# x = tf.ones((1,5,1))

# x_np[0,3,0]=1
# x = tf.convert_to_tensor(x_np,dtype=tf.float32)
# n = 3
# x_out= model_mv_tf(x,n)
# with tf.Session() as sess:
#     x_out_np = sess.run([x_out])
#     print(x_out_np)

## numpy version of moving average
def model_mv(x, n, isWeight=False):
    """
    for each batch, do the matrix level moving average
    :param: x: (b, t, d)
    :param n: lags (previous timesteps) in moving average
    :param isWeight: whether apply weight in mv
    :return: pred: (b,t,d)
    """
    def mv_matrix(x, n, isWeight):
        """
        matrix moving average, and keep the first n-1 vectors
        eg. x=[[1,1],[2,2],[3,3],[4,4]], n=3
        return [[1,1],[2,2],[2,2],[3,3]]
        gap: T*1 vector for time gap, gap[0]=0
        :param x: (t,d)
        :param n: lags (previous timesteps) in moving average
        :param isWeight: whether apply weight in mv
        :return: y: (t,d)
        """
        n = min(n, x.shape[0])
        if isWeight:
            y = np.copy(x)
            weight = np.exp(-np.arange(n - 1, -1, -1) / (n / 3.0))
            weight = weight / weight.sum()
            for i in range(n - 1, x.shape[0]):  # n=1: does not change
                y[i] = np.dot(weight, x[i - n + 1:i + 1])
            return y
        else:
            y = np.cumsum(x, dtype=np.float, axis=0)
            y[n:] = y[n:] - y[:-n]
            conv = y[n - 1:] / n
            no_conv = x[:n - 1]
            return np.concatenate((no_conv, conv), axis=0)

    y = np.zeros(x.shape)
    n_batch = x.shape[0]
    for i in range(n_batch):
        y[i] = mv_matrix(x[i], n, isWeight)
    return y





def model_mv_gap(x, x_gap, n, args=args):
    """
    for each batch, do the matrix moving average
    :param x: (b,t,d)
    :param x_gap: (b,t,1)
    :param args: object that contains all setup parameters (import from args.py)
    :return: pred (b,t,d)
    """
    def mv_matrix_gap(x, x_gap, n, args=args):
        """
        :param x: (t,d)
        :param x_gap: (t,1) time gap
        :param args: object that contains all setup parameters (import from args.py)
        :return: pred: (t,d)
        """
        n = min(n, x.shape[0])
        pred = np.copy(x)
        x_gap = np.squeeze(x_gap)
        for i in range(n - 1, x.shape[0]):
            gap = x_gap[i - n + 1:i + 1]  # eg, gap=[1,2,3]
            gap = gap[1:]  # eg, gap=[2,3]
            gap = np.append(gap, 0)  # gap=[2,3,0]
            gap_reverse = gap[::-1]  # gap_reverser =[0,3,2]
            weight = np.cumsum(gap_reverse)  # weight=[0,3,5]
            weight = weight[::-1]  # weight=[5,3,0]
            # weight = np.arange(n - 1, -1, -1)
            # weight = np.zeros((n,))
            weight = np.exp(-weight / args.gap_bandwidth)
            weight = weight / weight.sum()
            pred[i] = np.dot(weight, x[i - n + 1:i + 1])
        return pred
    pred = np.zeros(x.shape)
    n_batch = x.shape[0]
    for i in range(n_batch):
        pred[i] = mv_matrix_gap(x[i], x_gap[i], n)
    return pred

def model_mv_xing(x, n):
    """
    for each batch, do the matrix level moving average
    :param: x: (b, t, d)
    :param n: lags (previous timesteps) in moving average
    :param isWeight: whether apply weight in mv
    :return: pred: (b,t,d)
    """
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i, 0] = np.random.randint(1, np.amax(x) + 1)
        x_temp = x[i]
        x_temp = x_temp[x_temp>0]
        # print(x_temp.shape)
        for j in range(1,len(x_temp)+1):
            # start = max(0, j - n)
            start = 0
            out = stats.mode(x_temp[start:j], axis=-1)
            y[i,j] = out[0]
    return y

# x = np.array([[0,2,2,2,3,4,5,5,0,0],[0,1,1,43,1,43,54,0,0,0]])
# y = model_mv_xing(x,n=10)
# print(x)
# print(y)
# pdb.set_trace()

