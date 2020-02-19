import tensorflow as tf
from args import *


def calc_loss(pred,y,y_impression=None):
    """
    compute different types of loss
    l2: (pred -y)^2,
    hinge: \sum_y' max{0, f(<pred,y'>) - f(<pred,y>)}, f can be 'linear', 'sigmoid', 'log_sigmoid'
                y' is neg sample (impression), <x,y> is inner product
    bpr: -log(sigmoid(<pred,y'> - <pred,y>))
    nce: -log(sigmoid(<pred,y>)) - \sum_y' log(sigmoid(- <pred,y'>)) /k
    :param pred: (b,t,d)
    :param y: (b,t,d)
    :param y_impression: (b,t,k,d)
    :return: loss: (b,t)
    """
    if args.loss == 'l2':
        loss = tf.reduce_sum(tf.square(pred - y), 2)  # b,t
    if args.loss == 'cross_entropy':
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred)  # b,t
    elif args.loss == 'nce':
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.sigmoid(
            tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3))))  # [B,T,1,dim]*[B,T,dim,1]->[B,T,1,1]->[B,T]
        part_1 = tf.log(inner)  # [B,T]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
        part_2 = tf.reduce_sum(tf.log(tf.sigmoid(-inner_prod)), axis=2)  # [B,T]
        # loss = -part_1-part_2 # [B,T]
        loss = -part_1 - part_2 / args.num_neg_sample * args.nce_weight
    elif args.loss == 'hinge_sigmoid':
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3)),axis=3)  # [B,T,1,dim]*[B,T,dim,1]->[B,T,1]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
        part_1 = tf.sigmoid(inner)  # [B,T,1]
        part_2 = tf.sigmoid(inner_prod)  # [B,T,k]
        diff = part_2 - part_1 + args.hinge_delta  # [B,T,k]
        loss = tf.reduce_mean(tf.nn.relu(diff), axis=2)  # [B,T], avg of max
        # loss = tf.nn.relu(tf.reduce_max(diff,axis=2)) # [B,T], max of max
    elif args.loss == 'hinge_logsigmoid':
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3)),axis=3)  # [B,T,1,dim]*[B,T,dim,1]->[B,T,1]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
        part_1 = tf.log(tf.sigmoid(inner))  # [B,T,1]
        part_2 = tf.log(tf.sigmoid(inner_prod))  # [B,T,k]
        diff = part_2 - part_1 + args.hinge_delta  # [B,T,k]
        loss = tf.reduce_mean(tf.nn.relu(diff), axis=2)  # [B,T], avg of max
        # loss = tf.nn.relu(tf.reduce_max(diff,axis=2)) # [B,T], max of max
    elif args.loss == 'hinge_linear':
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3)),axis=3)  # [B,T,1,dim]*[B,T,dim,1]->[B,T,1]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
        part_1 = inner  # [B,T,1]
        part_2 = inner_prod  # [B,T,k]
        diff = part_2 - part_1 + args.hinge_delta  # [B,T,k]
        loss = tf.reduce_mean(tf.nn.relu(diff), axis=2)  # [B,T], avg of max
        # loss = tf.nn.relu(tf.reduce_max(diff,axis=2)) # [B,T], max of max
    elif args.loss == 'bpr':
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3)),axis=3)  # [B,T,1,dim]*[B,T,dim,1]->[B,T,1]
        part_1 = tf.sigmoid(inner)  # [B,T,1]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
        part_2 = tf.sigmoid(inner_prod)  # [B,T,k]
        diff = part_1 - part_2  # [B,T,k]
        loss = -tf.reduce_mean(tf.log(tf.sigmoid(diff)), axis=2)  # [B,T], avg of sigmoid diff
        # loss = -tf.nn.relu(tf.reduce_max(tf.log(tf.sigmoid(diff)),axis=2)) # max of max

    return loss

## loss and metric
def calc_score(pred, y_impression):
    """
    for each y' in y_impresssion, compute score(pred, y'); three modes: l2 (default), inner_prod, loss
    :param pred (b,t)
    :param y_impression (b,t,k)
    :return: score (b,t,k)
    """
    ## calc eval
    # find real embedding
    # y_id = tf.argmax(tf.reduce_min(tf.cast(tf.equal(y_impression, tf.expand_dims(y,2)),tf.int32),axis=-1), axis=-1, output_type=tf.int32) # b*t
    # get mask for y_impression

    # calc dist
    # inner product
    # y_impression_transpose = tf.transpose(y_impression, [0, 1, 3, 2])
    # inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), y_impression_transpose), axis=2) # inner product -> b*t*1*k -> b*t*k
    # l2
    if args.rank_metric=='l2':
        score = -tf.reduce_sum(tf.square(tf.expand_dims(pred, 2)-y_impression), axis=-1) # inner product -> b*t*1*k -> b*t*k
    if args.rank_metric=='inner_prod':
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [B,T,k,dim]->[B,T,dim,k]
        score = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [B,T,1,dim]*[B,T,dim,k]->[B,T,1,k]->[B,T,k]
    elif args.rank_metric=='loss':
        for i in range(args.max_impression_len):
            if i == 0:
                score = -tf.expand_dims(calc_loss(pred, y_impression[:,:,i,:]), axis=-1)
            else:
                score = tf.concat((score, -tf.expand_dims(calc_loss(pred, y_impression[:,:,i,:]), axis=-1)), axis=-1)

    return score


def calc_metric(score,mask_y,activity_count,user_count,y_id=None):
    """
    :param score: (b,t,k)
    :param mask_y: (b,t)
    :param mask_impression: (b,t,k)
    :param activity_count: (b,)
    :param user_count: scalar
    :return:
    scalar metrics: recall@1,recall@5,recall@10, mrr(mean reciprocal rank), mrp(mean rank percentile)
    ranks_float (b,t) for visualization
    """
    # rank dist
    ranks_values, ranks_indices = tf.nn.top_k(score, k=tf.shape(score)[-1])
    print(ranks_indices.get_shape(),'ranks_indices')
    # find the rank of the ground truth
    # ranks = tf.cast(tf.argmax(tf.cast(tf.equal(ranks_indices, 0), tf.int32), axis=-1), tf.float32) # b,t
    ranks = tf.cast(tf.argmax(tf.cast(tf.equal(ranks_indices, tf.expand_dims(y_id,axis=-1)), tf.int32), axis=-1), tf.float32) # b,t
    # impression_count = tf.reduce_sum(mask_impression,axis=-1) + tf.constant(1e-6) # b,t
    # impression_count = tf.cast(tf.shape(score)[-1],tf.float32) # b,t
    impression_count = args.item_num
    ranks_float = ranks / impression_count # b,t
    reciprocal_ranks_float = 1.0 / (1 + ranks) # b,t
    # recall@K, K in {1,5,10}

    recall1 = tf.where(ranks<=0, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t
    recall5 = tf.where(ranks<=4, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t
    recall10 = tf.where(ranks<=9, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t
    print(recall1.shape)

    # apply mask to remove invalid statistic
    ranks *= mask_y
    ranks_float *= mask_y
    impression_count *= mask_y
    reciprocal_ranks_float *= mask_y
    recall1 *= mask_y
    recall5 *= mask_y
    recall10 *= mask_y

    # get mean rank/rank_float/impression count for each user (mean over activity count)
    ranks_float_mean = tf.reduce_sum(ranks_float, 1) / activity_count # b

    reciprocal_ranks_float_mean = tf.reduce_sum(reciprocal_ranks_float, 1) / activity_count # b
    recall1_mean = tf.reduce_sum(recall1, 1)/activity_count # b
    recall5_mean = tf.reduce_sum(recall5, 1)/activity_count # b
    recall10_mean = tf.reduce_sum(recall10, 1)/activity_count # b

    ranks_float_mean_user = tf.reduce_sum(ranks_float_mean) / user_count # 1
    reciprocal_ranks_float_mean_user = tf.reduce_sum(reciprocal_ranks_float_mean) / user_count # 1
    recall1_mean_user = tf.reduce_sum(recall1_mean) / user_count # 1
    recall5_mean_user = tf.reduce_sum(recall5_mean) / user_count # 1
    recall10_mean_user = tf.reduce_sum(recall10_mean) / user_count # 1

    return recall1_mean_user, recall5_mean_user, recall10_mean_user, reciprocal_ranks_float_mean_user, ranks_float_mean_user, ranks_float, ranks_indices,ranks_values,ranks


def calc_metric_fast(score,mask_y,activity_count,user_count,y):
    """
    :param score: (b,t,k)
    :param mask_y: (b,t)
    :param mask_impression: (b,t,k)
    :param activity_count: (b,)
    :param user_count: scalar
    :return:
    scalar metrics: recall@1,recall@5,recall@10, mrr(mean reciprocal rank), mrp(mean rank percentile)
    ranks_float (b,t) for visualization
    """
    # rank dist
    if 'mv' in args.model_type:
        ranks = tf.reduce_sum(
            tf.cast(tf.greater(score, tf.reduce_sum(score * y, axis=-1, keep_dims=True)), dtype=tf.float32), axis=-1)*(args.item_num//2)
    else:
        ranks = tf.reduce_sum(tf.cast(tf.greater(score,tf.reduce_sum(score*y,axis=-1,keep_dims=True)),dtype=tf.float32),axis=-1)
    # ranks = tf.reduce_sum(tf.cast(tf.greater_equal(score,tf.reduce_sum(score*y,axis=-1,keep_dims=True)),dtype=tf.float32),axis=-1)

    # ranks_values, ranks_indices = tf.nn.top_k(score, k=tf.shape(score)[-1])
    # print(ranks_indices.get_shape(),'ranks_indices')
    # find the rank of the ground truth
    # ranks = tf.cast(tf.argmax(tf.cast(tf.equal(ranks_indices, 0), tf.int32), axis=-1), tf.float32) # b,t
    # ranks = tf.cast(tf.argmax(tf.cast(tf.equal(ranks_indices, tf.expand_dims(y_id,axis=-1)), tf.int32), axis=-1), tf.float32) # b,t
    # impression_count = tf.reduce_sum(mask_impression,axis=-1) + tf.constant(1e-6) # b,t
    # impression_count = tf.cast(tf.shape(score)[-1],tf.float32) # b,t
    impression_count = args.item_num
    ranks_float = ranks / impression_count # b,t
    reciprocal_ranks_float = 1.0 / (1 + ranks) # b,t
    # recall@K, K in {1,5,10}

    recall1 = tf.where(ranks<=0, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t
    recall5 = tf.where(ranks<=4, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t
    recall10 = tf.where(ranks<=9, 1*tf.ones(tf.shape(ranks)), 0*tf.ones(tf.shape(ranks))) # b,t

    # apply mask to remove invalid statistic
    ranks *= mask_y
    ranks_float *= mask_y
    impression_count *= mask_y
    reciprocal_ranks_float *= mask_y
    recall1 *= mask_y
    recall5 *= mask_y
    recall10 *= mask_y

    # get mean rank/rank_float/impression count for each user (mean over activity count)
    ranks_float_mean = tf.reduce_sum(ranks_float, 1) / activity_count # b

    reciprocal_ranks_float_mean = tf.reduce_sum(reciprocal_ranks_float, 1) / activity_count # b
    recall1_mean = tf.reduce_sum(recall1, 1)/activity_count # b
    recall5_mean = tf.reduce_sum(recall5, 1)/activity_count # b
    recall10_mean = tf.reduce_sum(recall10, 1)/activity_count # b

    ranks_float_mean_user = tf.reduce_sum(ranks_float_mean) / user_count # 1
    reciprocal_ranks_float_mean_user = tf.reduce_sum(reciprocal_ranks_float_mean) / user_count # 1
    recall1_mean_user = tf.reduce_sum(recall1_mean) / user_count # 1
    recall5_mean_user = tf.reduce_sum(recall5_mean) / user_count # 1
    recall10_mean_user = tf.reduce_sum(recall10_mean) / user_count # 1

    return recall1_mean_user, recall5_mean_user, recall10_mean_user, reciprocal_ranks_float_mean_user, ranks_float_mean_user, ranks_float, ranks



def calc_loss_gmm(pred,y,y_impression,pi,mu,sigma,gmm):
    """
    compute difference types of loss for the GMM model
    gmm_prob: compute the likelihood of obeserving the data
    gmm_prob_max: comput the prob that y belongs to each gaussian, and pick the max
    gmm_prob_hinge: maximize the likelihood of obeserving the data (y) and minize the likelihood of y_impression
    The above three loss does not require a predciton (y_hat) since it is a generative model, the following loss
    requries a y_hat
    gmm_hinge_mix: combine the hinge_loss (requires a prediction y_hat= mean of the gaussian ) with gmm_prob
    :param pred: (b,t,d)
    :param y: (b,t,d)
    :param y_impression: (b,t,k,d)
    :param mask_y: (b,t)
    :param pi: (b,t,m), m = number of gaussian in the mixture; pi is the prior of each gaussian class
    :param mu: (b,t,m,d) mean for each gaussian
    :param sigma: (b,t,m,d) std for each gaussian (the variance matrix is diagonal)
    :param gmm: gaussian mixture model object
    :return:
    loss: (b,t)
    """
    ## loss only for gmm model
    if args.loss == 'gmm_prob':
        loss = -tf.log(tf.exp(gmm.log_prob(y) / args.output_dim))  # b,t
    elif args.loss == 'gmm_prob_max':
        loss_list = []  # m*[b,t]
        for i in range(args.num_mixture):
            if not args.sigma:
                mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mu[:, :, i, :], scale_diag=sigma[:, :, i, :])
            else:
                mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mu[:, :, i, :], scale_diag=sigma)
            loss_list.append(pi[:, :, i] * tf.exp(mvn.log_prob(y) / args.output_dim))  # b,t
        loss = -tf.log(tf.reduce_max(tf.stack(loss_list, axis=-1), axis=-1))
    elif args.loss == 'gmm_prob_hinge':
        pos_prob = tf.log(tf.exp(gmm.log_prob(y) / args.output_dim))  # b,t
        # neg_sample_reform = tf.transpose(y_impression, [2, 0, 1, 3])  # [b,t,k,d]->[k,b,t,d]
        # neg_prob = tf.log(tf.exp(gmm.log_prob(neg_sample_reform) / args.output_dim))  # k,b,t
        for i in range(args.num_neg_sample):
            if i == 0:
                neg_prob = tf.expand_dims(tf.log(tf.exp(gmm.log_prob(y_impression[:,:,i,:])/args.output_dim)),axis=0) # 1,b,t
            else:
                neg_prob = tf.concat([neg_prob,tf.expand_dims(tf.log(tf.exp(gmm.log_prob(y_impression[:,:,i,:])/args.output_dim)),axis=0)], axis=0)
        diff = neg_prob - tf.expand_dims(pos_prob, axis=0) + args.hinge_delta # k,b,t
        loss = tf.reduce_mean(tf.nn.relu(diff), axis=0) # b,t

    elif args.loss == 'gmm_prob_hinge_logsigmoid':
        pos_prob = tf.log(tf.exp(gmm.log_prob(y) / args.output_dim))  # b,t
        for i in range(args.num_neg_sample):
            neg_prob = tf.log(tf.exp(gmm.log_prob(y_impression[:, :, i, :]) / args.output_dim))  # b,t
            if i == 0:
                loss = tf.nn.relu(tf.log(tf.sigmoid(neg_prob)) - tf.log(tf.sigmoid(pos_prob)) + args.hinge_delta)  # b,t
            else:
                loss = (loss * i + tf.nn.relu(
                    tf.log(tf.sigmoid(neg_prob)) - tf.log(tf.sigmoid(pos_prob)) + args.hinge_delta)) / (i + 1)  # b,t

    elif args.loss == 'gmm_hinge_mix' and args.pred_mode == 'gmm_mean':
        # hinge loss
        pred = tf.nn.l2_normalize(pred, dim=-1)
        inner = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), tf.expand_dims(y, 3)),axis=3)  # [b,t,1,d]*[b,t,d,1]->[b,t,1]
        neg_sample_reform = tf.transpose(y_impression, [0, 1, 3, 2])  # [b,t,k,dim]->[b,t,dim,k]
        inner_prod = tf.squeeze(tf.matmul(tf.expand_dims(pred, 2), neg_sample_reform),axis=2)  # [b,t,1,d]*[b,t,d,k]->[b,t,1,k]->[b,t,k]
        part_1 = tf.log(tf.sigmoid(inner))  # [b,t,1]
        part_2 = tf.log(tf.sigmoid(inner_prod))  # [b,t,k]
        diff = part_2 - part_1 + args.hinge_delta  # [b,t,k]
        hinge_loss = tf.reduce_mean(tf.nn.relu(diff), axis=2)  # [b,t], avg of max
        mle_loss = -tf.log(tf.exp(gmm.log_prob(y) / args.output_dim))
        # mle + hinge
        loss = args.mix_weight * mle_loss + hinge_loss

    return loss