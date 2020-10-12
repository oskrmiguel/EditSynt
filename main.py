#!/usr/bin/env python
# coding:utf8

from __future__ import print_function

import os
import argparse
import collections
import logging

import numpy as np
import torch
import torch.nn as nn

import data
import vocabulary
from checkpoint import Checkpoint
from editnts import EditNTS
from evaluator import Evaluator

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

def sort_by_lens(seq, seq_lengths):
    seq_lengths_sorted, sort_order = seq_lengths.sort(descending=True)
    seq_sorted = seq.index_select(0, sort_order)
    return seq_sorted, seq_lengths_sorted, sort_order

def reweigh_batch_loss(target_id_bath):
    pad_c = 0
    unk_c = 0
    keep_c = 0
    del_c = 0
    start_c = 0
    stop_c = 0
    other_c = 0

    new_edits_ids_l = target_id_bath
    for i in new_edits_ids_l:
        # start_c += 1
        # stop_c += 1
        for ed in i:
            if ed == PAD_ID:
                pad_c += 1
            elif ed == UNK_ID:
                unk_c += 1
            elif ed == KEEP_ID:
                keep_c += 1
            elif ed == DEL_ID:
                del_c += 1
            elif ed == START_ID:
                start_c +=1
            elif ed == STOP_ID:
                stop_c +=1
            else:
                other_c += 1

    NLL_weight = np.zeros(30006) + (1 / other_c+1)
    NLL_weight[PAD_ID] = 0  # pad
    NLL_weight[UNK_ID] = 1. / unk_c+1
    NLL_weight[KEEP_ID] = 1. / keep_c+1
    NLL_weight[DEL_ID] = 1. / del_c+1
    NLL_weight[5] = 1. / stop_c+1
    NLL_weight_t = torch.from_numpy(NLL_weight).float().cuda()
    # print(pad_c, unk_c, start_c, stop_c, keep_c, del_c, other_c)
    return NLL_weight_t

def reweight_global_loss(w_add,w_keep,w_del):
    # keep, del, other, (0, 65304, 246768, 246768, 2781648, 3847848, 2016880) pad,start,stop,keep,del,add
    NLL_weight = np.ones(30006)+w_add
    NLL_weight[PAD_ID] = 0  # pad
    NLL_weight[KEEP_ID] = w_keep
    NLL_weight[DEL_ID] = w_del
    return NLL_weight

def training(edit_net,nepochs, args, vocab):
    eval_dataset = data.Dataset(os.path.join(args.data_dir, 'val.df.filtered.pos')) # load eval dataset
    evaluator = Evaluator(loss= nn.NLLLoss(ignore_index=vocab.w2i['PAD'], reduction='none'), batch_size = args.batch_size)
    editnet_optimizer = torch.optim.Adam(edit_net.parameters(),
                                          lr=1e-3, weight_decay=1e-6)
    # scheduler = MultiStepLR(abstract_optimizer, milestones=[20,30,40], gamma=0.1)
    # abstract_scheduler = ReduceLROnPlateau(abstract_optimizer, mode='max')

    # uncomment this part to re-weight different operations
    # NLL_weight = reweight_global_loss(args.w_add, args.w_keep, args.w_del)
    # NLL_weight_t = torch.from_numpy(NLL_weight).float().cuda()
    # editnet_criterion = nn.NLLLoss(weight=NLL_weight_t, ignore_index=vocab.w2i['PAD'], reduce=False)
    editnet_criterion = nn.NLLLoss(ignore_index=vocab.w2i['PAD'], reduction='none')

    best_eval_loss = 999 # init statistics
    best_eval_sari = 0
    best_epoch = 0 # best epoch
    best_checkN = 0 # best check
    checkN = 0
    print_loss = []  # Reset every print_every

    for epoch in range(nepochs):
        # scheduler.step()
        #reload training for every epoch
        if os.path.isfile(os.path.join(args.data_dir, 'train.df.filtered.pos')):
            train_dataset = data.Dataset(os.path.join(args.data_dir, 'train.df.filtered.pos'))
        else:  # iter chunks and vocab_data
            train_dataset = data.Datachunk(os.path.join(args.data_dir, 'train.df.filtered.pos'))

        for i, batch_df in train_dataset.batch_generator(batch_size=args.batch_size, shuffle=True):

            #     time1 = time.time()
            prepared_batch, syn_tokens_list = data.prepare_batch(batch_df, vocab, args.max_seq_len, args.do_gcn) #comp,scpn,simp

            # a batch of complex tokens in vocab ids, sorted in descending order
            org_ids = prepared_batch[0]
            org_lens = org_ids.ne(0).sum(1)
            org = sort_by_lens(org_ids, org_lens)  # inp=[inp_sorted, inp_lengths_sorted, inp_sort_order]
            # a batch of pos-tags in pos-tag ids for complex
            org_pos_ids = prepared_batch[1]
            org_pos_lens = org_pos_ids.ne(0).sum(1)
            org_pos = sort_by_lens(org_pos_ids, org_pos_lens)

            out = prepared_batch[2][:, :]
            tar = prepared_batch[2][:, 1:]

            simp_ids = prepared_batch[3]
            adj = prepared_batch[4]

            editnet_optimizer.zero_grad()
            output = edit_net(org, out, org_ids, org_pos, adj, simp_ids)
            ##################calculate loss
            tar_lens = tar.ne(0).sum(1).float()
            tar_flat=tar.contiguous().view(-1)
            loss = editnet_criterion(output.contiguous().view(-1, vocab.count), tar_flat).contiguous()
            loss[tar_flat == 1] = 0 #remove loss for UNK
            loss = loss.view(tar.size())
            loss = loss.sum(1).float()
            loss = loss/tar_lens
            loss = loss.mean()

            print_loss.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(edit_net.parameters(), 1.)
            editnet_optimizer.step()

            if i % args.print_every == 0:
                log_msg = 'Epoch: %d, Step: %d, Loss: %.4f' % (
                    epoch,i, np.mean(print_loss))
                print_loss = []
                print(log_msg)

                # Checkpoint
            if i % args.check_every == 0:
                checkN += 1
                edit_net.eval()

                # import cProfile, pstats, io
                # pr = cProfile.Profile()
                # pr.enable()
                val_loss, bleu_score, sari, sys_out = evaluator.evaluate(eval_dataset, vocab, edit_net,args)
                # pr.disable()
                # s = io.StringIO()
                # sortby = 'cumulative'
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print(s.getvalue())
                log_msg = "epoch %d, step %d, Dev loss: %.4f, Bleu score: %.4f, Sari: %.4f \n" % (epoch, i, val_loss, bleu_score, sari)
                print(log_msg)

                # if val_loss < best_eval_loss:
                #     best_eval_loss = val_loss
                #     Checkpoint(model=edit_net,
                #                opt=editnet_optimizer,
                #                epoch=epoch, step=i,
                #     ).save(args.store_dir)
                if args.best_sari and sari > best_eval_sari:
                    best_epoch = epoch
                    best_checkN = checkN
                    best_eval_sari = sari
                    Checkpoint(model=edit_net,
                               opt=editnet_optimizer,
                               epoch=epoch, step=i,
                    ).save(args.store_dir)
                elif not args.best_sari and val_loss < best_eval_loss:
                    best_epoch = epoch
                    best_checkN = checkN
                    best_eval_loss = val_loss
                    Checkpoint(model=edit_net,
                               opt=editnet_optimizer,
                               epoch=epoch, step=i,
                    ).save(args.store_dir)
                print("checked after %d steps"%i)
                if args.early_stopping > 0 and (checkN - best_checkN) > args.early_stopping:
                    print("Early stopping (best epoch is {})".format(best_epoch))
                    return edit_net
                edit_net.train()
    return edit_net

def evaluation(infile, edit_net, args, vocab, outfile):
    edit_net.eval()
    eval_dataset = data.Dataset(infile) # load eval dataset
    evaluator = Evaluator(loss= nn.NLLLoss(ignore_index=vocab.w2i['PAD'], reduction='none'), batch_size = args.batch_size)
    val_loss, bleu_score, sari, sys_out = evaluator.evaluate(eval_dataset, vocab, edit_net, args)
    print("Bleu score: {:.4f}, Sari: {:.4f}".format(bleu_score, sari))
    if outfile is not None:
        print('Writing output in {}'.format(outfile))
        with open(outfile, "w",encoding="utf-8") as f:
            f.write('\n'.join(sys_out))
            f.write('\n')

def main():
    torch.manual_seed(233)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,dest='data_dir',
                        help='Directory with train/val data.')
    parser.add_argument('--store_dir', action='store', dest='store_dir',
                        help='Path to store models.')
    parser.add_argument('--postag_file', type=str, dest='postag_file',
                        default='vocab_path/ptb_ud_tagset.txt',
                        help='POS tag set file.')
    parser.add_argument('--embed_file', type=str, dest='embed_file',
                        default='vocab_path/glove.6B.100d.txt',
                        help='Embedding file.')
    parser.add_argument('--vocab_file', type=str, dest='vocab_file',
                        default=None,
                        help='Vocabulary file')
    parser.add_argument('--load_model', type=str, dest='load_model',
                        default=None,
                        help='Path for loading pre-trained model for further training')
    parser.add_argument('--do_gcn', action="store_true", help='Use GCN layer in encoder using dependency trees.')

    parser.add_argument('--eval_input', dest='eval_input', type=str, default=None,
                        help='Input file to evaluate.')
    parser.add_argument('--eval_output', dest='eval_output', type=str, default=None,
                        help='If --eval_input, where to store system output. If null, do not store')
    parser.add_argument('--vocab_size', dest='vocab_size', default=30000, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--max_seq_len', dest='max_seq_len', default=100)
    parser.add_argument('--check_every', dest='check_every', type=int, default=500,
                        help='Number of batches until next validation check (and model saving)')
    parser.add_argument('--print_every', dest='print_every', type=int, default=100,
                        help='Number of batches until information is printed.')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early_stopping', type=int, default=0, help='Stop if measure in dev does not improve. Zero means no early stopping.')
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--best_sari', action="store_true", help='Select model according to best sari in dev.')
    parser.add_argument('--device', type=int, default=1,
                        help='select GPU')

    #train_file = '/media/vocab_data/yue/TS/editnet_data/%s/train.df.filtered.pos'%dataset
    # test='/media/vocab_data/yue/TS/editnet_data/%s/test.df.pos' % args.dataset
    args = parser.parse_args()

    if args.eval_input and args.load_model is None:
        print("Can't --eval_input without a pretrained model")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(args.device)

    # load vocab-related files and init vocab
    print('*'*10)
    vocab = vocabulary.Vocab()
    vocab.add_vocab_from_file(args.vocab_file, args.vocab_size)
    word_embed_size = vocab.add_embedding(gloveFile=args.embed_file)
    pos_vocab = vocabulary.POSvocab(args.postag_file) #load pos-tags embeddings
    print('*' * 10)

    print(args)
    print("generating config")
    hyperparams=collections.namedtuple(
        'hps', #hyper=parameters
        ['vocab_size', 'embedding_dim',
         'word_hidden_units', 'sent_hidden_units',
         'pretrained_embedding', 'word2id', 'id2word',
         'pos_vocab_size', 'pos_embedding_dim', 'do_gcn']
    )
    hps = hyperparams(
        vocab_size=vocab.count,
        embedding_dim=word_embed_size,
        word_hidden_units=args.hidden,
        sent_hidden_units=args.hidden,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.w2i,
        id2word=vocab.i2w,
        pos_vocab_size=pos_vocab.count,
        pos_embedding_dim=30,
        do_gcn=args.do_gcn
    )

    print('init editNTS model')
    edit_net = EditNTS(hps, n_layers=1)
    edit_net.cuda()

    if args.load_model is not None:
        print("load edit_net for further training")
        ckpt_path = args.load_model
        ckpt = Checkpoint.load(ckpt_path)
        edit_net = ckpt.model
        edit_net.cuda()
        edit_net.train()

    if args.eval_input is not None:
        evaluation(args.eval_input, edit_net, args, vocab, args.eval_output)
    else:
        training(edit_net, args.epochs, args, vocab)


if __name__ == '__main__':
    import os
    cwd = os.getcwd()
    print(cwd)

    main()
