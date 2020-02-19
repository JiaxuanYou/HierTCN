from fastparquet import ParquetFile
import pandas as pd
import pdb
import time
import random
import multiprocessing as mp
# import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from args import *



def load_xing(path,min_activity=10, max_activity=args.max_seq_len,min_item=50, max_item=1e7):
    try:
        df = pd.read_csv(path,sep='\t')
    except:
        print('data not found')
    # filter data by action type
    filter_lambda = lambda df: df['interaction_type'] != 4
    df = df.loc[filter_lambda(df)]

    # filter data by item
    df = df.sort_values(['item_id'])
    viewer_data = df.values
    data_mask_item = np.zeros(len(df)) # mask which data to keep
    item_list, item_start, item_len = np.unique(viewer_data[:,1], return_index=True,
                                                return_counts=True)
    item_table = np.concatenate(
        (item_start[:, np.newaxis], item_len[:, np.newaxis], item_list[:, np.newaxis]), axis=1)
    item_table = item_table[(item_table[:, 1] >= min_item) & (item_table[:, 1] <= max_item)]
    # get mask
    for i in range(item_table.shape[0]):
        data_mask_item[item_table[i, 0]:item_table[i, 0] + item_table[i, 1]] = 1
    viewer_data = viewer_data[data_mask_item.astype(bool),:]
    df = pd.DataFrame(data=viewer_data,columns=['user_id','item_id','interaction_type','created_at'])

    # filter data by viewer
    df = df.sort_values(['user_id', 'created_at'])
    viewer_data = df.values
    data_mask_user = np.zeros(len(df)) # mask which data to keep
    viewer_list, activity_start, activity_len = np.unique(viewer_data[:,0], return_index=True,
                                                          return_counts=True)
    viewer_table = np.concatenate(
        (activity_start[:, np.newaxis], activity_len[:, np.newaxis], viewer_list[:, np.newaxis]),axis=1)
    viewer_table = viewer_table[(viewer_table[:, 1] >= min_activity) & (viewer_table[:, 1] <= max_activity)]
    # get mask
    for i in range(viewer_table.shape[0]):
        data_mask_user[viewer_table[i,0]:viewer_table[i,0]+viewer_table[i,1]] = 1
    viewer_data = viewer_data[data_mask_user.astype(bool),:]

    item_list = np.unique(viewer_data[:, 1])
    item_hash = dict(zip(item_list, np.arange(item_list.shape[0])+1))

    for i in range(viewer_data.shape[0]):
        viewer_data[i,1] = item_hash[viewer_data[i,1]]

    # build user table
    viewer_list, activity_start, activity_len = np.unique(viewer_data[:,0], return_index=True,
                                                          return_counts=True)
    viewer_table = np.concatenate(
        (activity_start[:, np.newaxis], activity_len[:, np.newaxis], viewer_list[:, np.newaxis]), axis=1)

    # session segmentation
    session = np.zeros((viewer_data.shape[0],1))
    session_counter_all = 0
    event_per_session = []
    session_per_user = []
    for i in range(viewer_table.shape[0]):
        session_counter = 0
        j_old = viewer_table[i,0]+1
        for j in range(viewer_table[i,0]+1,viewer_table[i,0]+viewer_table[i,1]):
            if (viewer_data[j,3]-viewer_data[j-1,3])//1800>0:
                session_counter+=1
                session_counter_all+=1
                event_per_session.append(j-j_old)
                j_old = j
            session[j] = session_counter
        session_counter_all += 1
        event_per_session.append(j - j_old)
        session_per_user.append(session_counter+1)
    viewer_data = np.concatenate((viewer_data,session),axis=1)
    event_per_session = np.array(event_per_session)
    print('event_per_session',np.mean(event_per_session),np.std(event_per_session), session_counter_all)
    session_per_user = np.array(session_per_user)
    print('session_per_user',np.mean(session_per_user),np.std(session_per_user))
    print('event_per_user',np.mean(viewer_table[:,1]),np.std(viewer_table[:,1]))
    # pdb.set_trace()
    # plt.hist(viewer_table[:,1],range=[0,300],bins=200)
    # plt.show()
    # print(np.amax(viewer_data[:,1]),len(item_list))
    return viewer_table,viewer_data

# viewer_table, viewer_data = load_xing('dataset/interactions.csv')
# pdb.set_trace()

