import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io
import imageio
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy
import pdb
from args import *



# set up tensorboard
def variable_summaries(var,name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        tf.summary.scalar('mean', tf.reduce_mean(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        # tf.summary.histogram('norm', tf.norm(var,axis=-1))



def ranks_analysis_plot_all(ranks,ranks_gap,ranks_user,ranks_count,ranks_gap_count,ranks_user_count,prefix):
    '''

    :param ranks: numpy ranks, B*T
    :return: tf image summary
    '''
    plt.switch_backend('agg')

    poly_rank = 2
    plt.figure()
    # plt.plot(np.arange(len(ranks)),ranks)
    x = np.arange(len(ranks))[ranks>0]
    y = ranks[ranks>0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path+prefix+'/0.png', format='png',dpi=300)
    np.save(args.log_path+prefix+'/0_x.npy',x)
    np.save(args.log_path+prefix+'/0_y.npy',y)
    buf.seek(0)
    img0 = imageio.imread(buf.getvalue(),format='png')
    plt.close()

    plt.figure()
    # plt.plot(np.arange(0,2200,10), ranks_gap)
    x = np.arange(0,2200,10)[ranks_gap > 0]
    y = ranks_gap[ranks_gap > 0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks_gap")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path+prefix+'/1.png', format='png',dpi=300)
    np.save(args.log_path +prefix+ '/1_x.npy', x)
    np.save(args.log_path +prefix+ '/1_y.npy', y)
    buf.seek(0)
    img1 = imageio.imread(buf.getvalue(), format='png')
    plt.close()

    plt.figure()
    # plt.plot(np.arange(0,2200,10), ranks_gap)
    x = np.arange(args.max_seq_len+1)[ranks_user > 0]
    y = ranks_user[ranks_user > 0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks_user")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path + prefix + '/2.png', format='png', dpi=300)
    np.save(args.log_path + prefix + '/2_x.npy', x)
    np.save(args.log_path + prefix + '/2_y.npy', y)
    buf.seek(0)
    img2 = imageio.imread(buf.getvalue(), format='png')
    plt.close()

    plt.figure()
    # plt.plot(np.arange(len(ranks)),ranks)
    x = np.arange(len(ranks_count))[ranks_count > 0]
    y = ranks_count[ranks_count > 0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks_count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path+prefix+'/3.png', format='png',dpi=300)
    np.save(args.log_path +prefix+ '/3_x.npy', x)
    np.save(args.log_path +prefix+ '/3_y.npy', y)
    buf.seek(0)
    img3 = imageio.imread(buf.getvalue(), format='png')
    plt.close()

    plt.figure()
    # plt.plot(np.arange(0,2200,10), ranks_gap)
    x = np.arange(0, 2200, 10)[ranks_gap_count > 0]
    y = ranks_gap_count[ranks_gap_count > 0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks_gap_count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path+prefix+'/4.png', format='png',dpi=300)
    np.save(args.log_path +prefix+ '/4_x.npy', x)
    np.save(args.log_path +prefix+ '/4_y.npy', y)
    buf.seek(0)
    img4 = imageio.imread(buf.getvalue(), format='png')
    plt.close()

    plt.figure()
    # plt.plot(np.arange(0,2200,10), ranks_gap)
    x = np.arange(args.max_seq_len+1)[ranks_user_count > 0]
    y = ranks_user_count[ranks_user_count > 0]
    fit = np.polyfit(x, y, poly_rank)
    fit_fn = np.poly1d(fit)
    plt.plot(x, y, 'bo', x, fit_fn(x), '--k')
    plt.title("ranks_user_count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(args.log_path + prefix + '/5.png', format='png', dpi=300)
    np.save(args.log_path + prefix + '/5_x.npy', x)
    np.save(args.log_path + prefix + '/5_y.npy', y)
    buf.seek(0)
    img5 = imageio.imread(buf.getvalue(), format='png')
    plt.close()


    return np.concatenate((img0[np.newaxis,:,:,:],img1[np.newaxis,:,:,:],img2[np.newaxis,:,:,:],
                           img3[np.newaxis,:,:,:],img4[np.newaxis,:,:,:],img5[np.newaxis,:,:,:]),axis=0)






#
# def topic_plot(topic_list_epoch,mask_list_epoch):
#
#     batch = 0
#
#     topic_list=topic_list_epoch[batch] # t*[b*D]
#     topic = np.stack(topic_list, axis=1) # b,t,D
#     mask_list=mask_list_epoch[batch] # t*[b*1]
#     mask = np.stack(mask_list, axis=1) # b,t,D
#
#     user = 0
#     while not np.all(mask[user]) and user < mask.shape[0]:
#         user +=1
#
#     kl = []
#     for t in range(1,topic.shape[1]):
#         kl.append(entropy(topic[user,t,:], topic[user,t-1,:]))
#
#     plt.switch_backend('agg')
#     plt.figure()
#     plt.plot(np.arange(len(kl)), np.array(kl))
#     plt.title("topic evolution")
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img = imageio.imread(buf.getvalue(), format='png')
#     return img

# def topic_plot_all(topic_list_epoch,mask_list_epoch):
#     img1 = topic_plot(topic_list_epoch,mask_list_epoch)
#     img2 = topic_plot(topic_list_epoch,mask_list_epoch)
#     img3 = topic_plot(topic_list_epoch,mask_list_epoch)
#     img4 = topic_plot(topic_list_epoch,mask_list_epoch)
#     img = np.concatenate([img1,img2,img3,img4],axis=0)
#     return img

def emb_plot(pred,true,impression,num_impression_show=10,num_show=100):
    '''

    :param pred: T*512
    :param real: T*512
    :param impression: T*K*512
    :return:
    '''

    pca = PCA(n_components=2)
    pred_u = pca.fit_transform(pred)
    true_u = pca.fit_transform(true)
    impression_list = []
    for i in range(num_impression_show):
        impression_list.append(pca.fit_transform(impression[:, i, :]))

    num_show = min(num_show,pred.shape[0])
    time = np.arange(num_show)

    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred_u[:num_show, 0], pred_u[:num_show, 1], time, color='red')
    ax.scatter(true_u[:num_show, 0], true_u[:num_show, 1], time, color='blue')
    for i in range(num_impression_show):
        ax.scatter(impression_list[i][:num_show, 0], impression_list[i][:num_show, 1], time, color='0.7')
    plt.title("embedding")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = imageio.imread(buf.getvalue(), format='png')
    return img

def emb_plot_all(pred,true,impression):
    img1 = emb_plot(pred[0],true[0],impression[0],num_show=30)[np.newaxis,:,:,:]
    img2 = emb_plot(pred[0],true[0],impression[0],num_show=100)[np.newaxis,:,:,:]
    img3 = emb_plot(pred[1],true[1],impression[1],num_show=30)[np.newaxis,:,:,:]
    img4 = emb_plot(pred[1],true[1],impression[1],num_show=100)[np.newaxis,:,:,:]
    img = np.concatenate([img1,img2,img3,img4],axis=0)
    return img


a = np.concatenate((np.zeros((2,2)).astype(object),np.array([['a'],['a']])),axis=1)
a1 = np.concatenate((np.zeros((2,2)).astype(object),np.array([['a'],['a']])),axis=1)
b = np.zeros((2,1)).astype(object)

print(np.array_equal(a,a1))