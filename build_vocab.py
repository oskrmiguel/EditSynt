import os
import argparse
from data_preprocess import Spacy

def do(fname, spacy, W):
    for line in open(fname, encoding='utf-8', errors='surrogate'):
        for token in [x for x in spacy.tokenize(line.lower())]:
            W[token] = W.get(token, 0) + 1
    return W

#fnames= ['/tartalo01/users/ocumbicus001/corpora/simplext/test.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/test.simple', '/tartalo01/users/ocumbicus001/corpora/simplext/train.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/train.simple', '/tartalo01/users/ocumbicus001/corpora/simplext/valid.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/valid.simple']
#fnames= ['/tartalo01/users/ocumbicus001/corpora/simplext/test.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/test.simple', '/tartalo01/users/ocumbicus001/corpora/simplext/train.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/train.simple', '/tartalo01/users/ocumbicus001/corpora/simplext/valid.complex', '/tartalo01/users/ocumbicus001/corpora/simplext/valid.simple']

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*',
                    help='Files in the corpus.')
parser.add_argument('--lang', type=str, default='en-large',
                    help='language.')

args = parser.parse_args()
if not args.files:
    print("[E] empty files")
    exit(1)

lang=args.lang
spacy = Spacy(lang)
W = {}
for fname in args.files:
    do(fname, spacy, W)

for k,v in sorted(W.items(), key=lambda item: item[1], reverse=True):
    print("{} {}".format(k, v))
