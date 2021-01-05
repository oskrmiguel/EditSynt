import os
import argparse
import random
import data_preprocess as dp
import vocabulary

random.seed(233)

def do_file(c_in, s_in, word_vocab, pos_vocab, outfile, lang, do_dep, discard_identical):
    #print(do_dep, discard_identical)
    complex_fh=open(c_in)
    simple_fh=open(s_in)
    df = dp.process_raw_data(complex_fh, simple_fh, pos_vocab, lang, discard_identical = discard_identical, do_dep = do_dep)
    return dp.editnet_data_to_editnetID(df, word_vocab, outfile)

parser = argparse.ArgumentParser()
parser.add_argument('lang', type=str,
                    help='Language. Possible values: {}.'.format(','.join(dp.Spacy.l2m.keys())))
parser.add_argument('vocab_file', type=str,
                    help='Vocabulary file. Will be created if --create_vocab is set')
parser.add_argument('complex', type=str,
                    help='File with complex sentences')
parser.add_argument('simple', type=str,
                    help='File with simple sentences')
parser.add_argument('out_file', type=str,
                    help='output file')
parser.add_argument('--nodep', action='store_true',
                    help='Set if no dependencies')
parser.add_argument('--preserve_identical', action='store_true',
                    help='Set to preserve identical sentences (in test)')
parser.add_argument('--vocab_n', type=int, default=30000,
                    help='Vocabulary size.')

args = parser.parse_args()

lang=args.lang
do_dep = not args.nodep # changeme to do dependency parsing
vocab_file = args.vocab_file
word_vocab = vocabulary.Vocab()
word_vocab.add_vocab_from_file(vocab_file, args.vocab_n)
pos_vocab = vocabulary.POSvocab('vocab_data/ptb_ud_tagset.txt')

do_file(args.complex, args.simple, word_vocab, pos_vocab, args.out_file, lang, do_dep, not args.preserve_identical)

# split 90% train, 10% val
# train=df.sample(frac=0.9,random_state=233) #random state is a seed value
# train.to_pickle(os.path.join(out, "train.df.filtered.pos"))
# val = df[~df.index.isin(train.index)]
# val.to_pickle(os.path.join(out, "val.df.filtered.pos"))
