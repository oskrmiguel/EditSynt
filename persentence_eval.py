import argparse
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu

def eval_sentence(orig_sent, sys_sent, ref_sent):
    sari = corpus_sari(orig_sents=orig_sent,
                       sys_sents=sys_sent,
                       refs_sents=[ref_sent], lowercase=True, tokenizer='13a')
    bleu = corpus_bleu(sys_sents=sys_sent, force=True,
                       lowercase = True,
                       tokenizer='13a',
                       refs_sents=[ref_sent])
    return sari, bleu

def read_file(filename, maxlines=None):
    with open(filename, encoding="utf-8") as f:
        #lines = [x.strip() for x in f.readlines()]
        lines = f.readlines()
    if maxlines is not None:
        return lines
    else:
        return lines[:maxlines]

parser = argparse.ArgumentParser()
parser.add_argument('orig', type=str, default='',
                    help='Original (complex) file.')
parser.add_argument('sys', type=str, default='',
                    help='System output.')
parser.add_argument('ref', nargs='*',
                    help='Reference simplification(s).')
args = parser.parse_args()

orig_data = read_file(args.orig)
sys_data = read_file(args.sys)
ref_data = read_file(args.ref[0])

#print(f"id,sari,bleu")
for i,(src, sys, ref) in enumerate(zip(orig_data, sys_data, ref_data)):
    sari, bleu = eval_sentence([src], [sys], [ref])
    print(f"{sari}\t{bleu}")
