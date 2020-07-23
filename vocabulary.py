import pickle

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
    def __init__(self):
        self.word_list = [PAD, UNK, KEEP, DEL, START, STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab_from_file(self, vocab_file="../vocab_data/vocab.txt",vocab_size=30000):
        with open(vocab_file, "r") as f:
            for i,line in enumerate(f):
                if i >=vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile="path_for_glove_embedding", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    # if len(model) % 1000 == 0:
                        # print("processed %d vocab_data" % len(model))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

class POSvocab():
    def __init__(self,vocab_path='vocab_data/'):
        self.word_list = [PAD,UNK,START,STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        with open(vocab_path+'ptb_ud_tagset.txt') as f:
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
