import os
import sys
import pickle
import random
import data_preprocess as dp
import vocabulary
import pandas as pd

random.seed(233)

if len(sys.argv) == 4:
    out = sys.argv[1]
    complex = sys.argv[2]
    simple = sys.argv[3]
else:
    out = "data/wikilarge"
    complex = "../TurkCorpus/data-simplification/wikilarge/wiki.full.aner.ori.train.src"
    simple = "../TurkCorpus/data-simplification/wikilarge/wiki.full.aner.ori.train.dst"

def build_vocab(c, s, out):
    V = {}
    for f in [c, s]:
        for line in open(f):
            words = line.split()
            for w in words:
                wl = w.lower()
                V[wl] = V.get(wl, 0) + 1
    fo = open(out, 'w')
    for k, v in sorted(V.items(), key=lambda x: x[1], reverse=True):
        fo.write("{} {}\n".format(k, v))

if not(os.path.exists(out)):
    print("Error: out directory \'{}\' does not exist".format(sys.argv[1]))
    exit(1)

build_vocab(complex, simple, os.path.join(out,"vocab.txt"))

complex_fh=open(complex)
simple_fh=open(simple)

pos_vocab = vocabulary.POSvocab()
df=(dp.process_raw_data(complex_fh, simple_fh, pos_vocab))

word_vocab = vocabulary.Vocab()
word_vocab.add_vocab_from_file(os.path.join(out,"vocab.txt"), 30000)
df=dp.editnet_data_to_editnetID(df, word_vocab, os.path.join(out, "all.df.filtered.pos"))

# split 90% train, 10% val
train=df.sample(frac=0.9,random_state=233) #random state is a seed value
train.to_pickle(os.path.join(out, "train.df.filtered.pos"))
val = df[~df.index.isin(train.index)]
val.to_pickle(os.path.join(out, "val.df.filtered.pos"))
