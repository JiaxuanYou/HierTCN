import numpy as np
from args import *
from utils import *


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20

def model_max_pin(x, x_gap, impression, n=10, has_gap=args.has_gap, w=args.gap_bandwidth):
    """
    for each candicate pin, find the closest pin in past n selected pins,
    and compute the inner prod as the score for this candicate pin. Finally
    take the candidate pin with max score as the prediction
    :param x: (b,t,d)
    :param x_gap: (b,t,1)
    :param impression: (b,t,k,d)
    :param n: number of past pins
    :param has_gap: whether use x_gap to weight pins
    :param w: bandwidth of exp(-t/w)
    :return pred: (b,t,d)
    """
    def max_pin_matrix(x, x_gap, impression, n, has_gap, w):
        """
        matrix level: select the pin with max score from impression
        :param x: (t,d)
        :param x_gap: (t,1)
        :param impression: (t,k,d)
        :param n: number of past pins
        :param has_gap: whether use x_gap to weight pins
        :param w: bandwidth of exp(-t/w)
        :return pred: (t,d)
        """
        pred = np.zeros(x.shape)
        num_nonzero=np.sum(np.any(x, axis=1)) # t'
        for i in range(num_nonzero-1):
            if i <= n - 1:
                past = x[:i+1,:]
                if has_gap:
                    gap = x_gap[:i+1]
            else:
                past = x[i-n+1:i+1,:] # (n,d)
                if has_gap:
                    gap = x_gap[i-n+1:i+1]
            pool = impression[i] # (k,d)
            pool = pool[np.any(pool, axis=1)]  # (k',d) delete zero padding
            score = np.matmul(past,pool.T)  # (n,k')
            if has_gap:
                gap = gap[1:]
                gap = np.append(gap, 0)
                gap_reverse = gap[::-1]
                weight = np.cumsum(gap_reverse)
                weight = weight[::-1]
                weight = np.exp(-weight / w) # (n,)
                weight = weight[:,np.newaxis] # (n,1)
                score = weight*score # n,k
            score_max = np.max(score, axis=0)  # (1,k')
            idx = np.argmax(score_max)
            pred[i] = pool[idx]
        return pred

    pred = np.zeros(x.shape)
    for b in range(x.shape[0]):
        pred[b] = max_pin_matrix(x[b], x_gap[b], impression[b], n, has_gap, w)
    return pred