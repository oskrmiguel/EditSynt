import os
import sys
import pickle
import random
import data_preprocess as dp
import vocabulary
import pandas as pd

random.seed(233)


def build_vocab(files, out, lang):
    V = {}
    F = []
    spacy = dp.Spacy(lang)
    for x in files:
        F.append(x[1])
        F.append(x[2])
    for fname in F:
        for line in open(fname):
            for w in spacy.tokenize(line.rstrip().lower()):
                V[w] = V.get(w, 0) + 1
    fo = open(out, 'w')
    for k, v in sorted(V.items(), key=lambda x: x[1], reverse=True):
        fo.write("{} {}\n".format(k, v))

def do_file(split, c_in, s_in, word_vocab, pos_vocab, outfile, lang):
    complex_fh=open(c_in)
    simple_fh=open(s_in)
    df = dp.process_raw_data(complex_fh, simple_fh, pos_vocab, lang, discard_identical = split != 'test')
    return dp.editnet_data_to_editnetID(df, word_vocab, outfile)

files = [['test', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.test.src', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.test.dst'],
['train', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.train.src', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.train.dst'],
['dev', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.valid.src', '/sc01a7/sisx09/sx09a1/jirhizts/Corpus/SimplificationDatasets/wikilarge-martin_split/wikilarge/wiki.full.aner.ori.valid.dst']]

#files = [['t_all', 't1', 't2']]

out ='data/wikilarge'
vocab_file = os.path.join(out,"vocab.txt")

lang='en'
build_vocab(files, vocab_file, lang)
word_vocab = vocabulary.Vocab()
word_vocab.add_vocab_from_file(vocab_file, 30000)
pos_vocab = vocabulary.POSvocab('vocab_data/ptb_ud_tagset.txt')

for split,c,s in files:
    print('Processing ', split)
    do_file(split, c, s, word_vocab, pos_vocab, os.path.join(out, "{}.df.filtered.pos".format(split)), lang)

# split 90% train, 10% val
# train=df.sample(frac=0.9,random_state=233) #random state is a seed value
# train.to_pickle(os.path.join(out, "train.df.filtered.pos"))
# val = df[~df.index.isin(train.index)]
# val.to_pickle(os.path.join(out, "val.df.filtered.pos"))
