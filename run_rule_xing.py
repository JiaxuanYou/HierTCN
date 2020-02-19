from data_loader import *
from model import *
import subprocess
import os
import pdb
import time
import threading
import pickle
from utils import *

def evaluate_rule(args, loader, sess, prefix='test',warm_start=args.warm_start):
    # evaluate, no training needed
    time_test_load = 0
    time_test_run = 0
    batch_count = 0
    test_loss_epoch = 0
    ranks_float_mean_np_epoch = 0
    reciprocal_ranks_float_mean_np_epoch = 0
    recall1_mean_np_epoch = 0
    recall5_mean_np_epoch = 0
    recall10_mean_np_epoch = 0
    while True:
        time1 = time.time()
        batch_x, batch_y, info = loader.get_batch()
        time2 = time.time()
        batch_y_pred = model_mv_xing(batch_x, n=args.lag)

        feed_dict = {y_pred_id: batch_y_pred, y_id: batch_y, lr: args.learning_rate}
        if warm_start:
            # batch_mask_warmstart = np.greater(info[:, :, 2], 1527897540000)
            batch_mask_warmstart = np.greater(info[:, :, 3], 1445617130)
            feed_dict[mask_warmstart] = batch_mask_warmstart

        test_loss, summary_val, ranks_float_np, pred_np, ranks_float_mean_np, \
        reciprocal_ranks_float_mean_np, recall1_mean_np, recall5_mean_np, recall10_mean_np, ranks_np = sess.run(
            [loss, val_summary, ranks_float, pred, ranks_float_mean_user, reciprocal_ranks_float_mean_user,
             recall1_mean_user, recall5_mean_user, recall10_mean_user,ranks],
            feed_dict=feed_dict)
        time3 = time.time()
        test_loss_epoch += test_loss
        ranks_float_mean_np_epoch += ranks_float_mean_np
        reciprocal_ranks_float_mean_np_epoch += reciprocal_ranks_float_mean_np
        recall1_mean_np_epoch += recall1_mean_np
        recall5_mean_np_epoch += recall5_mean_np
        recall10_mean_np_epoch += recall10_mean_np
        # pdb.set_trace()

        time_test_load += time2 - time1
        time_test_run += time3 - time2

        batch_count += 1
        if batch_count % 20 == 0:
            with open(args.log_path + prefix + '/results.txt', 'a') as f:
                f.write('batch count {}:'.format(batch_count))
                f.write('loss_epoch {}, ranks_float_epoch_mean {}\n'.format(test_loss_epoch / batch_count,
                                                                            ranks_float_mean_np_epoch / batch_count))
                f.write('reciprocal_ranks_float_epoch_mean {}, recall1_epoch_mean {}\n'.format(
                    reciprocal_ranks_float_mean_np_epoch / batch_count, recall1_mean_np_epoch / batch_count))
                f.write('recall5_epoch_mean {}, recall10_epoch_mean {}\n'.format(recall5_mean_np_epoch / batch_count,
                                                                                 recall10_mean_np_epoch / batch_count))
        if loader.done:
            break

    test_loss_epoch /= batch_count
    ranks_float_mean_np_epoch /= batch_count
    reciprocal_ranks_float_mean_np_epoch /= batch_count
    recall1_mean_np_epoch /= batch_count
    recall5_mean_np_epoch /= batch_count
    recall10_mean_np_epoch /= batch_count

    summary = tf.Summary()
    summary.value.add(tag="loss_epoch", simple_value=test_loss_epoch)
    summary.value.add(tag="ranks_float_epoch_mean", simple_value=ranks_float_mean_np_epoch)
    summary.value.add(tag="reciprocal_ranks_float_epoch_mean", simple_value=reciprocal_ranks_float_mean_np_epoch)
    summary.value.add(tag="recall1_epoch_mean", simple_value=recall1_mean_np_epoch)
    summary.value.add(tag="recall5_epoch_mean", simple_value=recall5_mean_np_epoch)
    summary.value.add(tag="recall10_epoch_mean", simple_value=recall10_mean_np_epoch)
    return summary


def run_rule(args,loader_test,warm_start=args.warm_start):
    '''
    run rule based model, no training needed
    mv, max_pin
    :param args:
    :return:
    '''
    # Start training
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init)
        test_writer = tf.summary.FileWriter(args.log_path + 'test/', sess.graph, flush_secs=10)
        test_random_writer = tf.summary.FileWriter(args.log_path + 'test_random/', sess.graph, flush_secs=10)

        if args.test:
            summary = evaluate_rule(args,loader_test,sess,prefix='test',warm_start=warm_start)
            test_writer.add_summary(summary, 0)

        print("Step 0: Done! {}".format(args.name))