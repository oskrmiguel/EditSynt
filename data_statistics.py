#!/usr/bin/env python

# compute statistics from data.

import sys
import argparse
import pandas as pd
from data_preprocess import Spacy

def process_file(fname, spacy, sent_lens, dep_depths):
    print('Processing {}'.format(fname), file=sys.stderr)
    for n, line in enumerate(open(fname)):
        if not n % 100:
            print('{}'.format(n), end=' ', file=sys.stderr, flush=True)
        spacy_doc = spacy.analize(line.split('\t'))
        sent_lens.append(len(spacy_doc))
        dep_depths.append(spacy.depth(spacy_doc))
    print(file=sys.stderr, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('lang', type=str,
                    help='Language. Possible values: {}.'.format(','.join(Spacy.l2m.keys())))
parser.add_argument('files', nargs='*',
                    help='Files to process.')
args = parser.parse_args()

spacy = Spacy(args.lang)
sent_lens = []
dep_depths = []
for file in args.files:
    process_file(file, spacy, sent_lens, dep_depths)

for l,d in zip(sent_lens, dep_depths):
    print(f"{l}\t{d}")

# df = pd.DataFrame(zip(sent_lens, dep_depths), columns=['sent_lens', 'dep_depth'])
# print('Sentence lengh average: {} std: {}'.format(df['sent_lens'].mean(), df['sent_lens'].std()))
# print('Dependency depth average: {} std: {}'.format(df['dep_depth'].mean(), df['dep_depth'].std()))
# if args.output_csv:
#     df.to_csv(args.output_csv, index=False,header=False)
#     print('Stored in {}'.format(args.output_csv))
