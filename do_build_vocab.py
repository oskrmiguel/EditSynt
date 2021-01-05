import sys
import data_preprocess as dp

def build_vocab(F, out, lang):
    V = {}
    spacy = dp.Spacy(lang)
    for fname in F:
        for line in open(fname, encoding='utf-8', errors='surrogateescape'):
            for w in spacy.tokenize(line.rstrip().lower()):
                V[w] = V.get(w, 0) + 1
    fo = open(out, 'w')
    for k, v in sorted(V.items(), key=lambda x: x[1], reverse=True):
        fo.write("{} {}\n".format(k, v))

if len(sys.argv) < 4:
    print("Usage: build_vocab lang out_file input_files")
    print("lang is one of {}".format(','.join(dp.Spacy.l2m.keys())))
    exit(1)

print('Building vocabulary {}'.format(sys.argv[2]))
build_vocab(sys.argv[3:], sys.argv[2], sys.argv[1])
