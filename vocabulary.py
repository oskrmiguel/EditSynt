import os
import pickle
import numpy as np

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

class Vocab():
    def __init__(self, logging=print):
        self.word_list = [PAD, UNK, KEEP, DEL, START, STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        self.logging = logging

    def add_vocab_from_file(self, vocab_file="../vocab_data/vocab.txt",vocab_size=30000):
        with open(vocab_file, "r") as f:
            for i,line in enumerate(f):
                if i >=vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        self.logging("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1
        return self.count

    def add_embedding(self, gloveFile):
        self.logging("loading embeddings")
        if gloveFile == 'NONE':
            self.embedding = np.zeros(shape=(len(self.word_list), 30))
            for i in range(self.count):
                self.embedding[i] = np.random.rand(30)
            return 30
        with open(gloveFile, 'r', encoding='utf-8', errors='surrogateescape') as f:
            model = {}
            w_set = set(self.word_list)
            first_line = True
            for line in f:
                splitLine = line.strip().split(' ')
                if first_line:
                    embed_size = len(splitLine) - 1
                    if embed_size == 1:
                        # fastext and skipgram have a first line wih N, dim
                        continue
                    embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))
                    first_line = False
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    # if len(model) % 1000 == 0:
                        # print("processed %d vocab_data" % len(model))
        self.embedding = embedding_matrix
        self.logging("%d words out of %d has embeddings in the embedding file" % (len(model), len(self.word_list)))
        return embed_size

class POSvocab():
    def __init__(self,vocab_file):
        self.word_list = [PAD,UNK,START,STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        with open(vocab_file) as f:
            # postag_set is from NLTK
            tagdict = [l.strip() for l in f]

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

        for w in tagdict:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1
