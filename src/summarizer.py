#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs, load_models_abs
from train_extractive import train_ext, validate_ext, test_ext
from prepro import data_builder
import glob, os
from rouge import Rouge

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='./bert_data_new/cnndm')
    parser.add_argument("-model_path", default='./models/')
    parser.add_argument("-result_path", default='./results/cnndm')
    parser.add_argument("-temp_dir", default='./temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=800, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    # params for preprocessing
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

 
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./logs/cnndm.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    return args, device_id

if __name__ == '__main__':
    # args, device_id = init_args()
    device_id = -1
    args = argparse.Namespace(accum_count=1, 
                              alpha=0.95, 
                              batch_size=32, 
                              beam_size=5, 
                              bert_data_path='./bert_data/cnndm', 
                              beta1=0.9, 
                              beta2=0.999, 
                              block_trigram=True, 
                              dec_dropout=0.2, 
                              dec_ff_size=2048, 
                              dec_heads=8, 
                              dec_hidden_size=768, 
                              dec_layers=6, 
                              enc_dropout=0.2, 
                              enc_ff_size=512, 
                              enc_hidden_size=512, 
                              enc_layers=6, 
                              encoder='bert', 
                              ext_dropout=0.2, 
                              ext_ff_size=2048, 
                              ext_heads=8, 
                              ext_hidden_size=768, 
                              ext_layers=2, 
                              finetune_bert=True, 
                              generator_shard_size=32, 
                              gpu_ranks=[0], 
                              label_smoothing=0.1, 
                              large=False, 
                              load_from_extractive='', 
                              log_file='./logs/val_abs_bert_cnndm', 
                              lower=True, 
                              lr=1, 
                              lr_bert=0.002, 
                              lr_dec=0.002, 
                              max_grad_norm=0, 
                              max_length=70, 
                              max_pos=512, 
                              max_src_nsents=100, 
                              max_src_ntokens_per_sent=200, 
                              max_tgt_len=140, 
                              max_tgt_ntokens=500, 
                              min_length=10, min_src_nsents=3, 
                              min_src_ntokens_per_sent=5, 
                              min_tgt_ntokens=5, 
                              mode='test_text', 
                              model_path='./models/', 
                              optim='adam', 
                              param_init=0, 
                              param_init_glorot=True, 
                              recall_eval=False, 
                              report_every=1, 
                              report_rouge=False, 
                              result_path='./results/abs_bert_cnndm_sample', 
                              save_checkpoint_steps=5, 
                              seed=666, 
                              sep_optim=True, 
                              shard_size=2000, 
                              share_emb=False, 
                              task='abs', 
                              temp_dir='./temp', 
                              test_all=False, 
                              test_batch_size=500, 
                              test_from='./models/CNN_DailyMail_Abstractive/model_step_148000.pt', 
                              test_start_from=-1, 
                              train_from='', 
                              train_steps=1000, 
                              use_bert_basic_tokenizer=False, 
                              use_bert_emb=False, 
                              use_interval=True, 
                              visible_gpus='-1', 
                              warmup_steps=8000, 
                              warmup_steps_bert=8000, 
                              warmup_steps_dec=8000, 
                              world_size=1)

    print(args.task, args.mode) 

    r = Rouge()
    cp = args.test_from
    try:
    	step = int(cp.split('.')[-2].split('_')[-1])
    except:
    	step = 0

    predictor = load_models_abs(args, device_id, cp, step)

    all_files = glob.glob(os.path.join('./bert_data/cnndm', '*'))
    print('Files In Input Dir: ' + str(len(all_files)))
    for file in all_files:
        with open(file) as f:
            source=f.read().rstrip()

        data_builder.str_format_to_bert(source, args, './bert_data_test/cnndm.test.0.bert.pt') 
        args.bert_data_path= './bert_data_test/cnndm'
        tgt, time_used = test_text_abs(args, device_id, cp, step, predictor)

        # some postprocessing 

        sentences = tgt.split('<q>')
        sentences = [sent.capitalize() for sent in sentences]
        sentences = '. '.join(sentences).rstrip()
        sentences = sentences.replace(' ,', ',')
        sentences = sentences+'.'

        print("summary [{}]".format(sentences))
        print(r.get_scores(sentences, source, avg=True))
        print("time used {}".format(time_used))
