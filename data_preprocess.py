import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
#import data
#from nltk import pos_tag
import spacy
from label_edits import sent2edit

# This script contains the reimplementation of the pre-process steps of the dataset
# For the editNTS system to run, the dataset need to be in a pandas DataFrame format
# with columns ['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_tags', 'comp_pos_ids', edit_labels','new_edit_ids', 'comp_dep_tree]

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

class Spacy:
    def __init__(self, lang):
        l2m = { 'en' : 'en_core_web_sm',
                'es' : 'es_core_news_sm',
                'fr' : 'fr_core_news_sm',
                'it' : 'it_core_news_sm'}
        if lang not in l2m:
            print('Language "{}" not supported.'.format(lang))
            exit(1)
        self.nlp = spacy.load(l2m[lang])
        self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)

    def analize(self, sent):
        return self.nlp(" ".join(sent))

    def tokenize(self, sent):
        if type(sent) == list:
            sent = " ".join(sent)
        return [token.text for token in self.tokenizer(sent)]

    def _dep2str(self, root, dep):
        str = '({}:{}'.format(dep, root.text)
        for child in root.children:
            str += self._dep2str(child, child.dep_)
        str += ')'
        return str

    def dep2str(self, doc):
        'Create a parse string from a analyzed dependency tree'
        root = [token for token in doc if token.head == token][0]
        return self._dep2str(root, 'root')


    def doc2adj(self, doc):
        '''Directly produce adjacency list from spacy output'''
        def _doc2adj(root, parents):
            root_id = token_map[root.text]
            parents.append([root_id, root_id]) # self-loop
            for child in root.children:
                parents.append([root_id, token_map[child.text]])
                _doc2adj(child, parents)
        comp_tokens = [token.text for token in doc]
        root = [token for token in doc if token.head == token][0]
        token_map = {x: i for i, x in enumerate(comp_tokens)}
        parents = []
        _doc2adj(root, parents)
        # pstrkk = [[comp_tokens[a], comp_tokens[b]] for a, b in parents]
        # print(parents)
        # print(pstrkk)
        N = len(comp_tokens)
        edges = np.array(parents, dtype=np.int32)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(N, N), dtype=np.float32)
        return adj

################## tree2graph

def parse_tree(tree, comp_tokens):

    def parse_node(node):
        i = node.find(':')
        if i == -1:
            return node, ''
        return node[i+1:], node[:i]

    token_map = {x: i for i, x in enumerate(comp_tokens)}
    node_map = dict()
    parents = []
    pstack = []
    node = ""
    # ret_exp = ""
    for char in tree:
        if char == ' ':
            continue
        if char not in ('(', ')'):
            node += char
            continue
        if node:
            node_name, node_type = parse_node(node)
            counter = token_map.get(node_name, None)
            if counter is None:
                # ERROR
                print('WARNING: token {} not in sentence\n{}'.format(node_name, " ".join(comp_tokens)))
                counter = -1
            else:
                node_map[counter] = {'name': node_name, 'type': node_type, 'idx': counter}
                # ret_exp += str(counter)
                parents.append([counter, counter]) # self loop
                if pstack and pstack[-1] != -1:
                    parents.append([pstack[-1], counter])
            pstack.append(counter)
        if char == ')':
            pstack.pop()
        # ret_exp += char
        node = ""
    return node_map, parents

def tree2adj(tree, comp_tokens):
    tree_map, parents = parse_tree(tree, comp_tokens)
    N = len(comp_tokens)
    edges = np.array(parents, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    return adj

######################

def remove_lrb(sent_string):
    # sent_string = sent_string.lower()
    frac_list = sent_string.split('-lrb-')
    clean_list = []
    for phrase in frac_list:
        if '-rrb-' in phrase:
            clean_list.append(phrase.split('-rrb-')[1].strip())
        else:
            clean_list.append(phrase.strip())
    clean_sent_string =' '.join(clean_list)
    return clean_sent_string

def replace_lrb(sent_string):
    sent_string = sent_string.lower()
    # new_sent= sent_string.replace('-lrb-','(').replace('-rrb-',')')
    new_sent = sent_string.replace('-lrb-', '').replace('-rrb-', '')
    return new_sent


def process_raw_data(comp_txt, simp_txt, pos_vocab, lang, discard_identical, do_dep = False):
    comp_txt = [line.lower().split() for line in comp_txt]
    simp_txt = [line.lower().split() for line in simp_txt]
    if discard_identical:
        comp_txt,simp_txt=zip(*[(i[0],i[1]) for i in zip(comp_txt,simp_txt) if i[0] != i[1]])

    # df_comp = pd.read_csv('data/%s_comp.csv'%dataset,  sep='\t')
    # df_simp= pd.read_csv('data/%s_simp.csv'%dataset,  sep='\t')
    assert len(comp_txt) == len(simp_txt)
    df = pd.DataFrame()
    def add_edits(df):
        """
        :param df: a Dataframe at least contains columns of ['comp_tokens', 'simp_tokens']
        :return: df: a df with an extra column of target edit operations
        """
        comp_sentences = df['comp_tokens'].tolist()
        simp_sentences = df['simp_tokens'].tolist()
        pair_sentences = list(zip(comp_sentences,simp_sentences))
        print("Generating edits ...", end='', flush = True)
        edits_list = [sent2edit(l[0],l[1]) for l in pair_sentences] # transform to edits based on comp_tokens and simp_tokens
        print("done")
        df['edit_labels'] = edits_list
        return df

    def add_pos_dep(df, src_sentences, pos_vocab, spacy):
        if do_dep:
            print("POS and DEP tagging:", end='', flush = True)
        else:
            print("POS tagging:", end='', flush = True)
        comp_sentences = []
        pos_sentences = []
        dep_sentences = []
        dep_sentences_rows = []
        dep_sentences_cols = []
        for i, sent in enumerate(src_sentences):
            if not i % 10000:
                print(' {}'.format(i), end=' ', flush = True)
            spacy_doc = spacy.analize(sent)
            pos_sentence = [(token.text, token.pos_) for token in spacy_doc]
            comp_sentence = [x[0] for x in pos_sentence]
            comp_sentences.append(comp_sentence)
            pos_sentences.append(pos_sentence)
            if do_dep:
                tree_str = spacy.dep2str(spacy_doc)
                adj = spacy.doc2adj(spacy_doc)
                dep_sentences.append(tree_str)
                dep_sentences_rows.append(adj.row)
                dep_sentences_cols.append(adj.col)
        print('done')
        df['comp_tokens'] = comp_sentences
        df['comp_pos_tags'] = pos_sentences
        if do_dep:
            df['comp_dep_tree'] = dep_sentences
            df['comp_dep_rows'] = dep_sentences_rows
            df['comp_dep_cols'] = dep_sentences_cols
        pos_ids_list = []
        for i,psent in enumerate(pos_sentences):
            pos_ids = [pos_vocab.w2i[w[1]] if w[1] in pos_vocab.w2i.keys() else pos_vocab.w2i[UNK] for w in psent]
            pos_ids_list.append(pos_ids)
        df['comp_pos_ids'] = pos_ids_list
        return df

    spacy = Spacy(lang)
    df['simp_tokens'] = [spacy.tokenize(x) for x in simp_txt]
    df = add_pos_dep(df, comp_txt, pos_vocab, spacy)
    df = add_edits(df)
    return df

def editnet_data_to_editnetID(df,vocab, output_path):
    """
    TODO: Uppdate doc: dependency columns only if do_dep == True
    this function reads from df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids', 'comp_dep_tree']
    and add vocab ids for comp_tokens, simp_tokens, and edit_labels
    :param df: df.columns=['comp_tokens', 'simp_tokens', 'edit_labels','comp_pos_tags','comp_pos_ids', 'comp_dep_tree']
    :param output_path: the path to store the df
    :return: a dataframe with df.columns=['comp_tokens', 'simp_tokens', 'edit_labels',
                                            'comp_ids','simp_id','edit_ids',
                                            'comp_pos_tags','comp_pos_ids',
                                            'comp_dep_tree', 'comp_dep_rows', 'comp_dep_cols'])
    """
    out_list = []

    def prepare_example(example, vocab):
        """
        :param example: one row in pandas dataframe with feild ['comp_tokens', 'simp_tokens', 'edit_labels']
        :param vocab: vocab object for translation
        :return: inp: original input sentence,
        """
        comp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
        simp_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['simp_tokens']])
        edit_id = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['edit_labels']])
        return comp_id, simp_id, edit_id  # add a dimension for batch, batch_size =1

    print("Writing pandas dataframe ... ", end='', flush = True)
    do_dep = False
    for i,example in df.iterrows():
        if do_dep == False and 'comp_dep_tree' in example is not None:
            do_dep = True
        comp_id, simp_id, edit_id = prepare_example(example,vocab)
        ex=[example['comp_tokens'], comp_id,
            example['simp_tokens'], simp_id,
            example['edit_labels'], edit_id,
            example['comp_pos_tags'],example['comp_pos_ids']
         ]
        if do_dep:
            ex += [example['comp_dep_tree'], example['comp_dep_rows'], example['comp_dep_cols']]
        out_list.append(ex)
    columns= ['comp_tokens','comp_ids', 'simp_tokens','simp_ids',
              'edit_labels','new_edit_ids','comp_pos_tags','comp_pos_ids']
    if do_dep:
        columns += ['comp_dep_tree', 'comp_dep_rows', 'comp_dep_cols']
    outdf = pd.DataFrame(out_list, columns=columns)
    outdf.to_pickle(output_path)
    print('saved to %s'%output_path)
    return outdf
