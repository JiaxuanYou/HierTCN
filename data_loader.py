from fastparquet import ParquetFile
import pandas as pd
import pdb
import time
import random
import multiprocessing as mp
# import cPickle as pickle
import os
import numpy as np
from args import *
import matplotlib.pyplot as plt
import pickle
import mmap
import struct
import numpy as np
from data_prepare import *

def save_object(obj, filename):
    '''
    save object as pickle
    :param obj:
    :param filename:
    :return:
    '''
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)




class Dataloader_single_level_model_xing():
    '''
    Dataloader without queue and session parsing, for single level models
    '''

    def __init__(self, args,type='train'):
        '''
        :param args: The following will be used:
        args.warm_start, args.max_impression_len, args.filter_user, args.data_splits,
        args.train_path, args.train_cache_path, args.load_num_train,
        args.test_path, args.test_cache_path, args.load_num_test,
        args.batch_size, args.test_data_ratio,
        :param has_sig: if has image signature. When getting html visualization, set True
        :param random_impression: Get random pin as impression, rather than ground truth impression
        :param max_impression_len: Maximum impression number within a feedview. If a feedview has more impressions, do sampling
        :param train: Whether load from train or test data
        :param partition_min: the min data cache id to load
        :param partition_max: the max data chche id to load
        '''
        # settings
        self.args = args
        self.done = False
        # load data
        self.viewer_table,self.viewer_data = load_xing('dataset/interactions.csv')
        if type=='train':
            self.viewer_table = self.viewer_table[0:int(self.viewer_table.shape[0]*0.8),:]
        if type=='validate':
            self.viewer_table = self.viewer_table[int(self.viewer_table.shape[0]*0.8):int(self.viewer_table.shape[0]*0.9),:]
        if type=='test':
            self.viewer_table = self.viewer_table[int(self.viewer_table.shape[0]*0.9):,:]
        print(type,'user num',self.viewer_table.shape[0],'item num',np.amax(self.viewer_data[:,1])+1,
              'activity num',self.viewer_data.shape[0])
        self.refresh()
        # plt.hist(self.viewer_data[:,3])
        # plt.show()
        # time_min = np.amin(self.viewer_data[:,3])
        # time_max = np.amax(self.viewer_data[:,3])
        # print(time_max,time_min)
        # print((time_max-time_min)*0.8+time_min)
        # pdb.set_trace()
    def refresh(self):
        if args.shuffle:
            self.index = np.random.permutation(self.viewer_table.shape[0])
        else:
            self.index = np.arange(self.viewer_table.shape[0])
        self.index_pointer = 0

    def get_batch(self):
        # select a batch of user
        viewerid_list = self.index[self.index_pointer:self.index_pointer + args.batch_size]

        # get the max sequence length
        seq_len = self.viewer_table[viewerid_list, 1].astype(int)
        max_len = np.amax(seq_len) + 1

        x = np.zeros((args.batch_size, max_len))  # activity data
        info = np.zeros((args.batch_size, max_len, 5))  # info for data batch
        for i, viewerid in enumerate(viewerid_list):
            # get activity data
            start = int(self.viewer_table[viewerid, 0])
            end = int(self.viewer_table[viewerid, 0]) + int(self.viewer_table[viewerid, 1])
            for j in range(start, end):
                x[i, j - start + 1] = int(self.viewer_data[j, 1])  # get activity data
                info[i, j - start + 1, :] = self.viewer_data[j, :]

        y = x[:, 1:]
        x = x[:, :-1]
        info = info[:, 1:, :]
        # update pointer
        self.done = False
        self.index_pointer += args.batch_size
        if self.index_pointer >= self.index.shape[0]:
            self.refresh()
            self.done = True

        return x, y, info


# loader = Dataloader_single_level_model_xing(args)
# x,y,info = loader.get_batch()
# pdb.set_trace()t



class Dataloader_hier_model_xing():
    def __init__(self,args,type='train'):
        '''
        Queue list is a list of queues: len=b, Q = [q1, q2, ...]
        Each queue is a list of sessions, variable length: q1 = [s1, s2, ...]
        Each session is a list of activities, variable length: s1 = [a1,a2,...]

        patition_max: excluded, one above max id
        :param args: The following will be used:
        args.warm_start, args.max_impression_len, args.filter_user, args.data_splits,
        args.train_path, args.train_cache_path, args.load_num_train,
        args.test_path, args.test_cache_path, args.load_num_test,
        args.batch_size, args.test_data_ratio,

        :param has_sig: if has image signature. When getting html visualization, set True
        :param random_impression: Get random pin as impression, rather than ground truth impression
        :param max_impression_len: Maximum impression number within a feedview. If a feedview has more impressions, do sampling
        :param train: Whether load from train or test data
        :param partition_min: the min data cache id to load
        :param partition_max: the max data chche id to load
        '''
        # settings
        self.args = args
        self.done = False
        # load data
        self.viewer_table,self.viewer_data = load_xing('dataset/interactions.csv')
        if type=='train':
            self.viewer_table = self.viewer_table[0:int(self.viewer_table.shape[0]*0.8),:]
        if type=='validate':
            self.viewer_table = self.viewer_table[int(self.viewer_table.shape[0]*0.8):int(self.viewer_table.shape[0]*0.9),:]
        if type=='test':
            self.viewer_table = self.viewer_table[int(self.viewer_table.shape[0]*0.9):,:]
        print(type,'user num', self.viewer_table.shape[0], 'item num', np.amax(self.viewer_data[:, 1]) + 1,
              'activity num', self.viewer_data.shape[0])

        self.refresh()

        # queue
        self.data = [[] for _ in range(args.batch_size)]  # b users -> k sessions -> m data points
        self.info = [[] for _ in range(args.batch_size)]  # b users -> k sessions -> m data points
        self.mask = [[] for _ in range(args.batch_size)]  # b users -> k sessions -> m data points
        self.queue_len = np.zeros(args.batch_size) # record the length of each queue
    def refresh(self,i=None):
        '''
        refresh data cache, load a new cache to the loader
        i: specify a data cache id to load.
        By default, i will be assigned so that it keeps looping over all the data
        :return: None
        '''
        if args.shuffle:
            self.index = np.random.permutation(self.viewer_table.shape[0])
        else:
            self.index = np.arange(self.viewer_table.shape[0])
        self.index_pointer = 0

    def enqueue(self):
        '''
        enque all the data of a new user
        :param store:
        :return:
        '''
        # get the start and end address of a user
        viewer_id = self.index[self.index_pointer]
        start = int(self.viewer_table[viewer_id, 0])
        end = int(self.viewer_table[viewer_id, 0]) + int(self.viewer_table[viewer_id, 1])
        session_info = self.viewer_data[start:end, 4]
        session_count = len(np.unique(session_info))

        # get data placeholders
        data_temp = [[] for _ in range(session_count)]
        info_temp = [[] for _ in range(session_count)]
        mask_temp = [1 for _ in range(session_count)]
        mask_temp[-1] = 0

        # parse data into sessions
        session_prev = self.viewer_data[start, 4]
        session_id = 0
        for j in range(start, end):
            session_current = self.viewer_data[j, 4]
            if session_current != session_prev:
                session_id += 1
                session_prev = session_current
            # get data
            if len(data_temp[session_id])<args.max_activity_len:
                data_temp[session_id].append(int(self.viewer_data[j, 1]))  # 512 dim
                info_temp[session_id].append(self.viewer_data[j, :])  # 5 dim
        ## enqueue data
        # find a queue
        queue_id = np.argmin(self.queue_len)
        # enqueue
        self.data[queue_id].extend(data_temp)
        self.info[queue_id].extend(info_temp)
        self.mask[queue_id].extend(mask_temp)

        # update length
        self.queue_len[queue_id] += session_count

        # update pointer
        self.index_pointer+=1
        if self.index_pointer>=self.index.shape[0]:
            self.refresh()
            return True # stop token
        return False

    def enqueue_loop(self):
        '''
        keep enqueue, until the queue is full for dequeue
        :param store:
        :return:
        '''
        stop = False
        while True:
            stop_temp = self.enqueue()
            stop = stop or stop_temp # if detect stop at any step
            if np.amin(self.queue_len) > self.args.max_session_num:
                break
        return stop

    def dequeue(self):
        '''
        dequeue to get a data batch
        :return:
        '''
        x_out = []
        y_out = []
        info_out = []
        mask_out = []
        for i in range(args.max_session_num):  # for sessions
            session_len = [len(self.data[user_id][0]) for user_id in range(self.args.batch_size)]
            session_len_max = max(session_len) + 1
            # pad zero
            session_data_x = np.zeros((self.args.batch_size, session_len_max))

            info_data = np.zeros((self.args.batch_size, session_len_max, 5))
            mask_data = np.zeros((self.args.batch_size, 1))  # first action is 0
            for user_id in range(self.args.batch_size):  # for users
                act_num = len(self.data[user_id][0])
                session_data_x[user_id, 1:act_num + 1] = np.stack(self.data[user_id][0], axis=0)
                info_data[user_id, 1:act_num + 1, :] = np.stack(self.info[user_id][0], axis=0)
                mask_data[user_id, 0] = self.mask[user_id][0]
                # delete data, dequeue
                del self.data[user_id][0]
                del self.info[user_id][0]
                del self.mask[user_id][0]
            self.queue_len -= 1
            x_out.append(session_data_x[:, :-1])
            y_out.append(session_data_x[:, 1:])
            info_out.append(info_data[:, 1:, :])
            mask_out.append(mask_data)

        return x_out,y_out,mask_out,info_out



    def get_batch(self):
        self.done = self.enqueue_loop() # report the stop token
        result = self.dequeue()
        return result


# loader = Dataloader_hier_model_xing(args)
# for i in range(1000):
#     x,y,mask,info = loader.get_batch()
# pdb.set_trace()

