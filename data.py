from collections import Counter

import glob
import random
import struct
import csv
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import torch
from torch.autograd import Variable
import random
import pickle
# <s> and </s> are used in the vocab_data files to segment the abstracts into sentences. They don't receive vocab ids.


PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

PAD_ID = 0 #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK_ID = 1 #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP_ID = 2 # This has a vocab id, which is used for copying from the source [2]
DEL_ID = 3 # This has a vocab id, which is used for deleting the corresponding word [3]
START_ID = 4 # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP_ID = 5 # This has a vocab id, which is used to stop decoding [5]

def log_batch(batch_indices, batch_tokens, logging = print, k=5):
    k = min(len(batch_indices), k)
    for i in range(k):
        logging(' '.join(['{}:{}'.format(token, idx) for idx, token in zip(batch_indices[i], batch_tokens[i])]))

def sent2id(sent,vocab):
    """
    this function transfers a sentence (in list of strings) to an np_array
    :param sent: sentence in list of strings
    :param vocab: vocab object
    :return: sentence in numeric token numbers
    """
    new_sent = np.array([[vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in sent]])
    return new_sent

def id2edits(ids,vocab):
    """
    #     this function transfers a id sentences of edits to actual edit actions
    #     :param ids: list of ids indicating edits
    #     :param vocab: vocab object
    #     :return: list of actual edits
    #     """
    edit_list = [vocab.i2w[i] for i in ids]
    return edit_list

def tokens2ids(df, vocab):
    unk = vocab.w2i[UNK]
    ids = []
    for s in df:
        id_array = np.array([vocab.w2i.get(i, unk) for i in s])
        ids.append(id_array)
    return ids

def batchify(data, max_len=100): #max_len cutout defined by human
    bsz = len(data)
    try:
        maxlen_data = max([s.shape[0] for s in data])
    except:
        maxlen_data = max([len(s) for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        try:
            batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        except:
            batch[i, :min(len(s), maxlen)] = s[:min(len(s), maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_start_stop(data, max_len=100): #max_len cutout defined by human
    # add start token at the beginning and stop token at the end of each sequence in a batch
    data = [np.append(s, [STOP_ID]) for s in data]  # stop 3
    data = [np.insert(s, 0, START_ID) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_stop(data, max_len=100): #max_len cutout defined by human
    # add stop tokens at the end of the sequence in each batch
    data = [np.append(s, [STOP_ID]) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()

def batchify_adj(rows, cols, tokens, max_len):
    max_n = max([s.shape[0] for s in tokens]) + 1
    max_len = min(max_n, max_len)
    adjs = []
    # for i, r in enumerate(rows):
    #     c = cols[i]
    for r,c in zip(rows, cols):
        adj = sp.coo_matrix((np.ones(r.shape[0]), (r, c)),
                            shape=(max_n, max_n), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.toarray()
        # divide by node degree
        asum = adj.sum(axis=1)
        asum[asum == 0] = 1
        adj = adj * np.diag(1.0 / asum)
        adjs.append(adj[:max_len, :max_len])
    return Variable(torch.from_numpy(np.array(adjs)).float()).cuda()

class Datachunk():
    def __init__(self,data_path):
        self.data_path = data_path
        self.listdir = os.listdir(self.data_path)
        random.shuffle(self.listdir)
        self.idx_count = 0

    def example_generator(self,shuffle=True):
        while len(self.listdir) != 0:
            print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())

            if shuffle:
                df = df.sample(frac=1, random_state=233).reset_index(drop=True)
                print('shuffling the df')

            for index, row in df.iterrows():
                self.idx_count+=1
                yield self.idx_count, row

    def batch_generator(self, batch_size=1, shuffle=True):
        while len(self.listdir) != 0:
            # print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())
            if shuffle:
                df = df.sample(frac=1, random_state=233).reset_index(drop=True)
                # print('shuffling the df')

            list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
            for df in list_df:
                self.idx_count += 1
                yield self.idx_count, df

class Dataset():
    def __init__(self,data_path):
        self.df = pd.read_pickle(data_path)
        self.idx_count = 0

    def example_generator(self):
        for index, row in self.df.iterrows():
            yield index, row

    def batch_generator(self, batch_size, shuffle=True):
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=233).reset_index(drop=True)
            # print('shuffling the df')

        list_df = [self.df[i:i + batch_size] for i in range(0, self.df.shape[0], batch_size)]
        for df in list_df:
            # if self.idx_count == 0:
            #     for i in range(0,min(len(df), 5)):
            #         print("***Example {} ***".format(i))
            #         print(df.iloc[i])
            self.idx_count += 1
            yield self.idx_count, df


def prepare_batch(batch_df,vocab, max_length=100, do_gcn = False):
    """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_ids', edit_labels','new_edit_ids']
        :param vocab: vocab object for translation
        :return: inp: original input sentences
        :return: inp_pos: pos-tag ids for the input sentences
        :return: adj: adjacency matrices of dependency trees from complex sentences
        :return: tgt: the target edit-labels in ids
        :return: inp_simp:the corresponding simple sentences in ids
        :return: batch_df['comp_tokens']:the complex tokens
        """
    inp = batchify_stop(tokens2ids(batch_df['comp_tokens'], vocab), max_len=max_length)
    inp_pos = batchify_stop(batch_df['comp_pos_ids'], max_len=max_length)
    inp_simp=batchify_start_stop(tokens2ids(batch_df['simp_tokens'], vocab), max_len=max_length)
    # tgt = batchify_start_stop(batch_df['edit_ids'], max_len=max_length)  # edit ids has early stop
    tgt = batchify_start_stop(tokens2ids(batch_df['edit_labels'], vocab), max_len=max_length)  # new_edit_ids do not do early stopping
    # I think new edit ids do not ave early stopping
    adj = None
    if do_gcn:
        if 'comp_dep_rows' in batch_df:
            adj = batchify_adj(batch_df['comp_dep_rows'], batch_df['comp_dep_cols'], batch_df['comp_ids'], max_len=max_length)
        else:
            print("ERROR: no dependency graph in data.")
            exit(1)
    return [inp, inp_pos, tgt, inp_simp, adj], batch_df['comp_tokens']
