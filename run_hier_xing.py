from data_loader import *
from model import *
from utils import *
import subprocess
import os
import pdb
import time
import threading
import pickle

def ranks_analysis(ranks_float_np, batch_x_list, batch_mask_y, ranks_temp, ranks_temp_count):
    '''
    analyze model performance, on each time step within session
    :param ranks_float_np: B*T
    :param batch_x_list: list of B*T*512, used to get session length
    :param batch_mask_y: B*T, used to get activity number
    :param ranks_temp: summed rank for time step i across all sessions
    :param ranks_temp_count: summed activity number for time step i acorss all sessions
    :return: updated ranks_temp,ranks_temp_count
    '''
    session_len = np.array([x.shape[1] for x in batch_x_list])
    session_len_max = np.amax(session_len)
    session_start = np.cumsum(np.concatenate((np.zeros(1),session_len[:-1]),axis=0))
    session_end = np.cumsum(session_len)-1
    for id in range(session_len_max): # loop over within session id
        for i in range(len(session_len)): # loop over sessions
            id_temp = int(session_start[i]+id)
            if id_temp <= session_end[i]:
                ranks_temp[id] += np.sum(ranks_float_np[:, id_temp])
                ranks_temp_count[id] += np.sum(batch_mask_y[:,id_temp])
    return ranks_temp,ranks_temp_count

def ranks_gap_analysis(ranks_float_np, batch_x_list, batch_mask_y, batch_x_gap_list, ranks_temp, ranks_temp_count):
    '''
    analyze gap performance, over different time dap across session
    :param ranks_float_np: B*T
    :param batch_x_list: list of B*T*512, used to get session length
    :param batch_mask_y: B*T, used to get activity number
    :param batch_x_gap_list: list of B, gap time between sessions
    :param ranks_temp: summed rank for binned time gap across all sessions
    :param ranks_temp_count: summed activity number for time step i acorss all sessions
    :return:
    '''
    session_len = np.array([x.shape[1] for x in batch_x_list])
    session_start = np.cumsum(np.concatenate((np.zeros(1),session_len[:-1]),axis=0))
    session_end = np.cumsum(session_len)-1

    gap_bins = np.arange(0,2200,10)
    for i in range(len(session_len)): # loop over sessions
        for j in range(args.batch_size): # loop over users
            gap_value = batch_x_gap_list[i][j]
            gap_id = np.digitize(gap_value,gap_bins)

            ranks_temp[gap_id] += np.sum(ranks_float_np[j, int(session_start[i]):int(session_end[i]+1)])
            ranks_temp_count[gap_id] += np.sum(batch_mask_y[j, int(session_start[i]):int(session_end[i]+1)])
    return ranks_temp,ranks_temp_count


def ranks_user_analysis(ranks_float_np, info_np,batch_mask_y, viewer_table, ranks_temp, ranks_temp_count):
    '''
    analyze performance, over different users: user sessions/activities number
    :param ranks_float_np: B*T
    :param batch_x_list: list of B*T*512, used to get session length
    :param batch_mask_y: B*T, used to get activity number
    :param info_np: information of users
    :param ranks_temp: summed rank for user activity length, len 1000 array
    :param ranks_temp_count: summed activity number for user activity length, len 1000 array
    :return:
    '''

    for i in range(args.batch_size):
        for j in range(ranks_float_np.shape[1]):
            viewer_id = int(info_np[i, j, -1])

            if viewer_id != 0 and viewer_id<viewer_table.shape[0]:
                viewer_activity_len = viewer_table[viewer_id,1]
                ranks_temp[viewer_activity_len] += ranks_float_np[i,j]
                ranks_temp_count[viewer_activity_len] += batch_mask_y[i,j]
    return ranks_temp,ranks_temp_count



def evaluate_hier(args, loader, state_np, sess, y_slice_np=None,topic_state_np=None,prefix='test',warm_start=args.warm_start):
    '''
    helper function for train_nn, doing evaluation for validation/test set.
    for single-level model
    :param args:
    :param loader: data loader
    :param state_np: pervious hidden states
    :param sess: tf session
    :param has_plot: if plot analysis results
    :return:
    '''
    # validate
    time_validate_load = 0
    time_validate_run = 0
    batch_count = 0 # counter for batch per epoch
    test_loss_epoch = 0 # init performance metric, same below
    ranks_float_mean_np_epoch = 0
    ranks_temp = np.zeros(args.max_activity_len + 1)
    ranks_temp_count = np.zeros(args.max_activity_len + 1)
    ranks_user_temp = np.zeros(args.max_seq_len+1)
    ranks_user_temp_count = np.zeros(args.max_seq_len+1)
    reciprocal_ranks_float_mean_np_epoch = 0
    recall1_mean_np_epoch = 0
    recall5_mean_np_epoch = 0
    recall10_mean_np_epoch = 0
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    # binary input
    if args.binary_input:
        pin_mean = np.load('/data1/home/ywang/pin_mean.npy')
    while True: # loop until loader.done is True
        time1 = time.time()
        viewer_table = loader.viewer_table.copy()
        batch_x_list, batch_y_list, batch_mask_list, info = loader.get_batch()
        batch_y = np.concatenate(batch_y_list, axis=1)
        time2 = time.time()


        feed_dict = {key: val for key, val in zip(x_id_list, batch_x_list)}
        feed_dict_y = {key: val for key, val in zip(y_id_list, batch_y_list)}
        feed_dict_mask = {key: val for key, val in zip(mask_list, batch_mask_list)}
        feed_dict.update(feed_dict_y)
        feed_dict.update(feed_dict_mask)
        if warm_start:
            info_concat = np.concatenate(info, axis=1)
            # batch_mask_warmstart = np.greater(info_concat[:, :, 2], 1527897540000)
            batch_mask_warmstart = np.greater(info_concat[:, :, 3], 1445617130)
            feed_dict[mask_warmstart] = batch_mask_warmstart

        feed_dict[y_id] = batch_y
        feed_dict[state_in] = state_np
        feed_dict[lr] = args.learning_rate
        feed_dict[is_train] = False

        # do evaluation, no train_op
        if args.model_type=='hier_slice':
            feed_dict[y_id_slice_in] = y_slice_np
            feed_dict[topic_state_in] = topic_state_np
            test_loss, summary_val, pred_np, state_np, y_slice_np, topic_state_np, \
            ranks_float_mean_user_np, ranks_float_np, mask_y_np, reciprocal_ranks_float_mean_np, recall1_mean_np, recall5_mean_np, recall10_mean_np = \
                sess.run([loss, val_summary, pred, state, y_slice, topic_state, ranks_float_mean_user, ranks_float, mask_y,
                     reciprocal_ranks_float_mean_user, recall1_mean_user, recall5_mean_user, recall10_mean_user],feed_dict,options=run_options)
        else:
            test_loss, summary_val, pred_np, state_np, ranks_float_mean_user_np, ranks_float_np, mask_y_np, \
            reciprocal_ranks_float_mean_np, recall1_mean_np, recall5_mean_np, recall10_mean_np = \
                sess.run([loss, val_summary, pred, state, ranks_float_mean_user, ranks_float, mask_y,
                          reciprocal_ranks_float_mean_user, recall1_mean_user, recall5_mean_user,
                          recall10_mean_user], feed_dict,options=run_options)
        # pdb.set_trace()
        # collect performance
        test_loss_epoch += test_loss
        ranks_float_mean_np_epoch += ranks_float_mean_user_np
        ranks_temp, ranks_temp_count = ranks_analysis(ranks_float_np, batch_x_list, mask_y_np, ranks_temp,
                                                      ranks_temp_count)
        ranks_user_temp, ranks_user_temp_count = ranks_user_analysis(ranks_float_np, np.concatenate(info, axis=1),mask_y_np,
                                                                   viewer_table,ranks_user_temp, ranks_user_temp_count)
        reciprocal_ranks_float_mean_np_epoch += reciprocal_ranks_float_mean_np
        recall1_mean_np_epoch += recall1_mean_np
        recall5_mean_np_epoch += recall5_mean_np
        recall10_mean_np_epoch += recall10_mean_np
        time3 = time.time()

        time_validate_load += time2 - time1
        time_validate_run += time3 - time2

        batch_count += 1
        # pdb.set_trace()

        if batch_count%20==0:
            with open(args.log_path + prefix + '/results.txt', 'a') as f:
                f.write('batch count {}:'.format(batch_count))
                f.write('loss_epoch {}, ranks_float_epoch_mean {}\n'.format(test_loss_epoch/batch_count, ranks_float_mean_np_epoch/batch_count))
                f.write('reciprocal_ranks_float_epoch_mean {}, recall1_epoch_mean {}\n'.format(
                    reciprocal_ranks_float_mean_np_epoch/batch_count, recall1_mean_np_epoch/batch_count))
                f.write('recall5_epoch_mean {}, recall10_epoch_mean {}\n'.format(recall5_mean_np_epoch/batch_count,recall10_mean_np_epoch/batch_count))
        if loader.done:
            break

    # calc per epoch performance
    test_loss_epoch /= batch_count
    ranks_float_mean_np_epoch /= batch_count
    ranks_temp /= ranks_temp_count + 1e-6
    ranks_user_temp /= ranks_user_temp_count + 1e-6
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
        f.write('final\n')
        f.write('loss_epoch {}, ranks_float_epoch_mean {}\n'.format(test_loss_epoch, ranks_float_mean_np_epoch))
        f.write('reciprocal_ranks_float_epoch_mean {}, recall1_epoch_mean {}\n'.format(reciprocal_ranks_float_mean_np_epoch, recall1_mean_np_epoch))
        f.write('recall5_epoch_mean {}, recall10_epoch_mean {}\n'.format(recall5_mean_np_epoch, recall10_mean_np_epoch))


    if args.model_type=='hier_slice':
        return summary, summary_val, time_validate_load, time_validate_run,state_np,y_slice_np,topic_state_np
    else:
        return summary, summary_val, time_validate_load, time_validate_run,state_np


def run_hier(args,loader_train,loader_validate,loader_test,warm_start=args.warm_start):
    '''
    train function for hier models, including all variants
    :param args:
    :param loader_train,loader_validate,loader_test,loader_test_random:
    :return:
    '''
    # Start training
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init)
        if args.load:
            try:
                fname = args.load_path + str(args.load_epoch) + ".ckpt"
                # Restore variables from disk.
                saver.restore(sess, fname)
                print("Model restored: {}".format(fname))
                epoch = args.load_epoch+1
            except:
                print("No ckpt found: {}, train from scratch".format(fname))
                epoch = 0
        else:
            epoch = 0
        # writer for tensorboard
        train_writer = tf.summary.FileWriter(args.log_path+'train/', sess.graph, flush_secs=30)
        validate_writer = tf.summary.FileWriter(args.log_path+'validate/', sess.graph, flush_secs=30)
        test_writer = tf.summary.FileWriter(args.log_path+'test/', sess.graph, flush_secs=30)

        # initialize user states
        # note that they will be carried over epochs, since each batch is cut from all user activites,
        # thus they can come from the same user, and we should keep the previous hidden states for the user
        # also note that each loader should have a separate hidden state, for the same reason
        state_np = np.zeros((args.batch_size, args.hidden_dim * args.num_layer))
        state_np_validate = np.zeros((args.batch_size, args.hidden_dim * args.num_layer))
        state_np_test = np.zeros((args.batch_size, args.hidden_dim * args.num_layer))

        if args.model_type=='hier_slice':
            # initialize y_slice and topic_state
            # that will carry over different batches
            y_slice_np = np.zeros((args.batch_size, args.output_dim))
            y_slice_np_validate = np.zeros((args.batch_size, args.output_dim))
            y_slice_np_test = np.zeros((args.batch_size, args.output_dim))

            topic_state_np = np.ones((args.batch_size, args.topic_dim)) / float(args.topic_dim)
            topic_state_np_validate = np.ones((args.batch_size, args.topic_dim)) / float(args.topic_dim)
            topic_state_np_test = np.ones((args.batch_size, args.topic_dim)) / float(args.topic_dim)

        # binary input
        if args.binary_input:
            pin_mean = np.load('/data1/home/ywang/pin_mean.npy')
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

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
                    batch_x_list, batch_y_list, batch_mask_list, info = loader_train.get_batch()
                    batch_y = np.concatenate(batch_y_list, axis=1)
                    time2 = time.time()

                    feed_dict = {key: val for key, val in zip(x_id_list, batch_x_list)}
                    feed_dict_y = {key: val for key, val in zip(y_id_list, batch_y_list)}
                    feed_dict_mask = {key: val for key, val in zip(mask_list, batch_mask_list)}
                    feed_dict.update(feed_dict_y)
                    feed_dict.update(feed_dict_mask)
                    if warm_start:
                        info_concat = np.concatenate(info, axis=1)
                        batch_mask_warmstart = np.less_equal(info_concat[:, :, 3], 1445617130) * np.greater(info_concat[:, :, 3], 0)
                        feed_dict[mask_warmstart] = batch_mask_warmstart
                    feed_dict[y_id] = batch_y
                    feed_dict[state_in] = state_np
                    feed_dict[lr] = args.learning_rate
                    feed_dict[is_train] = True

                    if args.model_type == 'hier_slice':
                        feed_dict[y_slice_in] = y_slice_np
                        feed_dict[topic_state_in] = topic_state_np
                        _, training_loss, summary_train, state_np, pred_np, y_slice_np, topic_state_np \
                            = sess.run([train_op, loss, train_summary, state, pred, y_slice, topic_state], feed_dict, options=run_options)
                    else:
                        _,training_loss,summary_train, state_np, pred_np = sess.run(
                            [train_op, loss, train_summary, state, pred], feed_dict, options=run_options)

                    training_loss_epoch += training_loss
                    time3 = time.time()
                    time_train_load += time2-time1
                    time_train_run += time3-time2

                training_loss_epoch /= args.epoch_batches_train

                summary = tf.Summary()
                summary.value.add(tag="loss_epoch", simple_value=training_loss_epoch)
                # train_writer.add_summary(summary_train, epoch)
                train_writer.add_summary(summary, epoch)

                # save model, don't save on the first epoch
                if args.save and epoch%args.save_epoch==0 and epoch!=0:
                    # Save the variables to disk.
                    fname = args.save_path+str(epoch)+".ckpt"
                    save_path = saver.save(sess, fname)
                    print("Model saved in path: %s" % save_path)

            # do evaluation on validation set and test set (optional: random impression set)
            if epoch % args.test_epoch==0 and epoch!=0 or args.train==False:
                # on validation set
                if args.validate:
                    if args.model_type == 'hier_slice':
                        summary, summary_val, time_validate_load, time_validate_run, state_np_validate,y_slice_np_validate,topic_state_np_validate = \
                            evaluate_hier(args, loader_validate, state_np_validate, sess,y_slice_np_validate,topic_state_np_validate,prefix='validate',warm_start=warm_start)
                    else:
                        summary, summary_val, time_validate_load, time_validate_run, state_np_validate = \
                            evaluate_hier(args, loader_validate, state_np_validate, sess,prefix='validate',warm_start=warm_start)
                    # validate_writer.add_summary(summary_val, epoch)
                    validate_writer.add_summary(summary, epoch)

                # on test set
                if args.test:
                    if args.model_type == 'hier_slice':
                        summary, summary_val, time_test_load, time_test_run, state_np_test,y_slice_np_test,topic_state_np_test = \
                            evaluate_hier(args, loader_test, state_np_test, sess,y_slice_np_test,topic_state_np_test,prefix='test',warm_start=warm_start)
                    else:
                        summary, summary_val, time_test_load, time_test_run, state_np_test = \
                            evaluate_hier(args, loader_test, state_np_test, sess,prefix='test',warm_start=warm_start)
                    # test_writer.add_summary(summary_val, epoch)
                    test_writer.add_summary(summary, epoch)

                    print("Step {}: test load {:.2f}s, test run {:.2f}s, {}".format(epoch,time_test_load,time_test_run,args.name))
                if args.train==False:
                    break # do evaluation once
            epoch += 1
