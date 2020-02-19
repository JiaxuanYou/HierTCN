from data_loader import *
from model import *
import subprocess
import os
import pdb
import time
import threading
import pickle

def evaluate_nn(args, loader, sess, prefix='test',warm_start=args.warm_start):
    '''
    helper function for train_nn, doing evaluation for validation/test set.
    for single-level model
    :param args:
    :param loader: data loader
    :param sess: tf session
    :return:
    '''
    time_validate_load = 0
    time_validate_run = 0
    batch_count = 0  # counter for batch per epoch
    test_loss_epoch = 0  # init performance metric, same below
    ranks_float_mean_np_epoch = 0
    reciprocal_ranks_float_mean_np_epoch = 0
    recall1_mean_np_epoch = 0
    recall5_mean_np_epoch = 0
    recall10_mean_np_epoch = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    # binary input
    if args.binary_input:
        pin_mean = np.load('/data1/home/ywang/pin_mean.npy')# 512 d vector
    while True:  # loop until loader.done is True
        time1 = time.time()
        batch_x, batch_y, info = loader.get_batch()
        time2 = time.time()

        feed_dict = {x_id: batch_x, y_id: batch_y,lr: args.learning_rate}
        if warm_start:
            # batch_mask_warmstart = np.greater(info[:, :, 2], 1527897540000)
            batch_mask_warmstart = np.greater(info[:, :, 3], 1445617130)
            feed_dict[mask_warmstart] = batch_mask_warmstart
        feed_dict[is_train] = False

        # do evaluation, no train_op
        test_loss, summary_val, ranks_float_np, pred_np, ranks_float_mean_user_np, \
        reciprocal_ranks_float_mean_np, recall1_mean_np, recall5_mean_np, recall10_mean_np = sess.run(
            [loss, val_summary, ranks_float, pred, ranks_float_mean_user, reciprocal_ranks_float_mean_user,
             recall1_mean_user, recall5_mean_user, recall10_mean_user], feed_dict=feed_dict,options=run_options)
        # pdb.set_trace()
        # collect performance
        test_loss_epoch += test_loss
        ranks_float_mean_np_epoch += ranks_float_mean_user_np
        reciprocal_ranks_float_mean_np_epoch += reciprocal_ranks_float_mean_np
        recall1_mean_np_epoch += recall1_mean_np
        recall5_mean_np_epoch += recall5_mean_np
        recall10_mean_np_epoch += recall10_mean_np
        time3 = time.time()

        time_validate_load += time2 - time1
        time_validate_run += time3 - time2

        batch_count += 1  # counter +1
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
    # calc per epoch performance
    test_loss_epoch /= batch_count
    ranks_float_mean_np_epoch /= batch_count
    reciprocal_ranks_float_mean_np_epoch /= batch_count
    recall1_mean_np_epoch /= batch_count
    recall5_mean_np_epoch /= batch_count
    recall10_mean_np_epoch /= batch_count

    summary = tf.Summary()
    summary.value.add(tag="loss_epoch", simple_value=test_loss_epoch)
    summary.value.add(tag="ranks_float_epoch_mean", simple_value=ranks_float_mean_np_epoch)
    summary.value.add(tag="reciprocal_ranks_float_epoch_mean",
                      simple_value=reciprocal_ranks_float_mean_np_epoch)
    summary.value.add(tag="recall1_epoch_mean", simple_value=recall1_mean_np_epoch)
    summary.value.add(tag="recall5_epoch_mean", simple_value=recall5_mean_np_epoch)
    summary.value.add(tag="recall10_epoch_mean", simple_value=recall10_mean_np_epoch)

    with open(args.log_path + prefix +'/results.txt', 'a') as f:
        f.write('loss_epoch {}, ranks_float_epoch_mean {}\n'.format(test_loss_epoch, ranks_float_mean_np_epoch))
        f.write('reciprocal_ranks_float_epoch_mean {}, recall1_epoch_mean {}\n'.format(reciprocal_ranks_float_mean_np_epoch, recall1_mean_np_epoch))
        f.write('recall5_epoch_mean {}, recall10_epoch_mean {}\n'.format(recall5_mean_np_epoch, recall10_mean_np_epoch))

    return summary,summary_val,time_validate_load,time_validate_run




def run_nn(args,loader_train,loader_validate,loader_test,warm_start=args.warm_start):
    '''
    training function for single-level model: gru, tcn
    :param args: argument object
    :param loader_train,loader_validate,loader_test: different data loaders
    :return:
    '''
    # Start training
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init)
        epoch=0
        if args.load:
            try:
                fname = args.load_path + str(args.load_epoch) + ".ckpt"
                # Restore variables from disk.
                saver.restore(sess, fname)
                print("Model restored: {}".format(fname))
                epoch = args.load_epoch+1 # recover epoch num
            except:
                print("No ckpt found: {}, train from scratch".format(fname))
                epoch = 0
        # writer for tensorboard
        train_writer = tf.summary.FileWriter(args.log_path+'train/', sess.graph, flush_secs=30)
        validate_writer = tf.summary.FileWriter(args.log_path+'validate/', sess.graph, flush_secs=30)
        test_writer = tf.summary.FileWriter(args.log_path+'test/', sess.graph, flush_secs=30)


        while epoch <= args.epoch_max:
            # if train the model
            if args.train:
                # learning rate schedule
                if args.lr_schedule:
                    if epoch==60:
                        args.learning_rate /= 5
                    if epoch==120:
                        args.learning_rate /= 5
                time_train_load = 0
                time_train_run = 0
                # train
                training_loss_epoch = 0
                for i in range(args.epoch_batches_train):
                    time1 = time.time()
                    batch_x, batch_y, info = loader_train.get_batch() # get data

                    feed_dict = {x_id: batch_x, y_id: batch_y, lr: args.learning_rate}
                    if warm_start:
                        # batch_mask_warmstart = np.less_equal(info[:, :, 2], 1527897540000) * np.greater(info[:, :, 2], 0)
                        batch_mask_warmstart = np.less_equal(info[:, :, 3], 1445617130) * np.greater(info[:, :, 3], 0)
                        feed_dict[mask_warmstart] = batch_mask_warmstart
                    feed_dict[is_train] = True
                    time2 = time.time()
                    _, training_loss, summary_train = sess.run([train_op,loss, train_summary], feed_dict=feed_dict,options=run_options)
                    training_loss_epoch += training_loss
                    time3 = time.time()
                    time_train_load += time2-time1
                    time_train_run += time3-time2

                training_loss_epoch /= args.epoch_batches_train # get average performance per epoch
                summary = tf.Summary()
                summary.value.add(tag="loss_epoch", simple_value=training_loss_epoch)
                # train_writer.add_summary(summary_train, epoch) # add result to tensorboard
                train_writer.add_summary(summary, epoch)

                # save model, don't save on the first epoch
                if args.save and epoch % args.save_epoch == 0 and epoch!=0:
                    # Save the variables to disk.
                    fname = args.save_path + str(epoch) + ".ckpt"
                    save_path = saver.save(sess, fname)
                    print("Model saved in path: %s" % save_path)

            # do evaluation on validation set and test set
            if epoch % args.test_epoch==0 and epoch!=0 or args.train==False:
                # on validation set
                if args.validate:
                    summary, summary_val, time_validate_load, time_validate_run = \
                        evaluate_nn(args, loader_validate, sess,prefix='validate',warm_start=warm_start)
                    # validate_writer.add_summary(summary_val, epoch)
                    validate_writer.add_summary(summary, epoch)

                # on test set
                if args.test:
                    summary, summary_val, time_test_load, time_test_run = \
                        evaluate_nn(args, loader_test, sess,prefix='test',warm_start=warm_start)
                    # test_writer.add_summary(summary_val, epoch)
                    test_writer.add_summary(summary, epoch)
                    print("Step {}: test load {:.2f}s, test run {:.2f}s {}".format(
                        epoch,time_test_load,time_test_run,args.name))
                if args.train==False:
                    break # do evaluation once
            epoch += 1
