# coding:utf-8
import torch
import numpy as np
import json
import opennre.encoder, opennre.model, opennre.framework
import sys
import os
import argparse
import random
import logging
from tqdm import tqdm
import openpyxl


def set_seed(seed=666):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='/home/zhangweiqi/.cache/huggingface/hub/bert-base-uncased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='default',
                    help='Checkpoint name')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')
parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
parser.add_argument('--seed', default=6666, type=int, help="random seed, default is 1")  # [1, 49, 1234, 2021, 4321]
# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=32, type=int,  # baseline:16
                    help='Batch size')
parser.add_argument('--lr', default=3e-5, type=float,
                    help='Learning rate')  # baseline:2e-5
parser.add_argument('--max_length', default=128, type=int,  # baseline:128
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=8, type=int,  # baseline:8
                    help='Max number of training epochs')
parser.add_argument('--sample_ratio', default=1.0, type=float,
                    help="only for low resource.")
args = parser.parse_args()


set_seed(args.seed)  # set seed, default is 1
# Some basic settings
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt_path = 'ckpt/re999/{}.pth.tar'.format(args.ckpt)

# text
root_path = '.'
args.train_file = os.path.join(root_path, 'data', 'txt/ours_train.txt')
args.val_file = os.path.join(root_path, 'data', 'txt/ours_val.txt')
args.test_file = os.path.join(root_path, 'data', 'txt/ours_test.txt')
# args.test_file = os.path.join(root_path, 'data', 'zeroshot_notintrain.txt.txt')
# args.test_file = os.path.join(root_path, 'data', 'zeroshot_nottoken.txt.txt')
# original image

# 原版
args.pic_train_file = os.path.join(root_path, 'data',  'img_org/train')
args.pic_val_file = os.path.join(root_path, 'data',  'img_org/val')
args.pic_test_file = os.path.join(root_path, 'data',  'img_org/test')

# ViT版本
args.vit_org_train_file = os.path.join(root_path, 'data',  'ViT_H5/vit_org_train.h5')
args.vit_org_val_file = os.path.join(root_path, 'data',  'ViT_H5/vit_org_val.h5')
args.vit_org_test_file = os.path.join(root_path, 'data',  'ViT_H5/vit_org_test.h5')
args.vit_vg_train_file = os.path.join(root_path, 'data',  'ViT_H5/vit_vg_train.h5')
args.vit_vg_val_file = os.path.join(root_path, 'data',  'ViT_H5/vit_vg_val.h5')
args.vit_vg_test_file = os.path.join(root_path, 'data',  'ViT_H5/vit_vg_test.h5')

# # CLIP版本
# args.clip_org_train_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_org_train.h5')
# args.clip_org_val_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_org_val.h5')
# args.clip_org_test_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_org_test.h5')
# args.clip_vg_train_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_vg_train.h5')
# args.clip_vg_val_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_vg_val.h5')
# args.clip_vg_test_file = os.path.join(root_path, 'data',  'CLIP_H5/clip_vg_test.h5')
#

# # head_entity_caption
# args.head_entity_train_file = os.path.join(root_path, 'data', 'head_entity_caption/entity_train.txt')
# args.head_entity_val_file = os.path.join(root_path, 'data', 'head_entity_caption/entity_val.txt')
# args.head_entity_test_file = os.path.join(root_path, 'data', 'head_entity_caption/entity_test.txt')
# # tail_entity_caption
# args.tail_entity_train_file = os.path.join(root_path, 'data', 'tail_entity_caption/entity_train.txt')
# args.tail_entity_val_file = os.path.join(root_path, 'data', 'tail_entity_caption/entity_val.txt')
# args.tail_entity_test_file = os.path.join(root_path, 'data', 'tail_entity_caption/entity_test.txt')

# # GPT3.5
# # head_entity_caption
# args.head_entity_train_file = os.path.join(root_path, 'data', 'gpt35-turbo/head_entity_train.txt')
# args.head_entity_val_file = os.path.join(root_path, 'data', 'gpt35-turbo/head_entity_val.txt')
# args.head_entity_test_file = os.path.join(root_path, 'data', 'gpt35-turbo/head_entity_test.txt')
# # tail_entity_caption
# args.tail_entity_train_file = os.path.join(root_path, 'data', 'gpt35-turbo/tail_entity_train.txt')
# args.tail_entity_val_file = os.path.join(root_path, 'data', 'gpt35-turbo/tail_entity_val.txt')
# args.tail_entity_test_file = os.path.join(root_path, 'data', 'gpt35-turbo/tail_entity_test.txt')

# GPT4mini
# head_entity_caption
args.head_entity_train_file = os.path.join(root_path, 'data', 'gpt4o-mini/head_entity_train.txt')
args.head_entity_val_file = os.path.join(root_path, 'data', 'gpt4o-mini/head_entity_val.txt')
args.head_entity_test_file = os.path.join(root_path, 'data', 'gpt4o-mini/head_entity_test.txt')
# tail_entity_caption
args.tail_entity_train_file = os.path.join(root_path, 'data', 'gpt4o-mini/tail_entity_train.txt')
args.tail_entity_val_file = os.path.join(root_path, 'data', 'gpt4o-mini/tail_entity_val.txt')
args.tail_entity_test_file = os.path.join(root_path, 'data', 'gpt4o-mini/tail_entity_test.txt')

# # GPT4mini_plus
# # head_entity_caption
# args.head_entity_train_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/head/entity_train.txt')
# args.head_entity_val_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/head/entity_val.txt')
# args.head_entity_test_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/head/entity_test.txt')
# # tail_entity_caption
# args.tail_entity_train_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/tail/entity_train.txt')
# args.tail_entity_val_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/tail/entity_val.txt')
# args.tail_entity_test_file = os.path.join(root_path, 'data', 'gpt4omini_Plus/tail/entity_test.txt')

# # llama70b
# # head_entity_caption
# args.head_entity_train_file = os.path.join(root_path, 'data', 'llama70b/head/entity_train.txt')
# args.head_entity_val_file = os.path.join(root_path, 'data', 'llama70b/head/entity_val.txt')
# args.head_entity_test_file = os.path.join(root_path, 'data', 'llama70b/head/entity_test.txt')
# # tail_entity_caption
# args.tail_entity_train_file = os.path.join(root_path, 'data', 'llama70b/tail/entity_train.txt')
# args.tail_entity_val_file = os.path.join(root_path, 'data', 'llama70b/tail/entity_val.txt')
# args.tail_entity_test_file = os.path.join(root_path, 'data', 'llama70b/tail/entity_test.txt')


# target relations
args.rel2id_file = os.path.join(root_path, 'data', 'ours_rel2id.json')
if not os.path.exists(args.test_file):
    logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
    args.test_file = args.val_file
args.metric = 'micro_f1'

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
id2rel = {v: k for k, v in rel2id.items()}

# 在这里接受由loader传出来的batch，然后把caption作为参数传给TMR_RE

# Define the sentence encoder
sentence_encoder = opennre.encoder.TMR_RE(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )

# Define the model
# torch.cuda.manual_seed_all(47)
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    train_pic_path=args.pic_train_file,
    train_head_ent_path=args.head_entity_train_file,  # 不用entity的时候注释掉
    train_tail_ent_path=args.tail_entity_train_file,  # 不用entity的时候注释掉
    train_vit_org_train=args.vit_org_train_file,  # vit
    train_vit_vg_train=args.vit_vg_train_file,  # vit
    # train_clip_org_train=args.clip_org_train_file,  # clip
    # train_clip_vg_train=args.clip_vg_train_file,  # clip
    val_path=args.val_file,
    val_pic_path=args.pic_val_file,
    val_head_ent_path=args.head_entity_val_file,  # 不用entity的时候注释掉
    val_tail_ent_path=args.tail_entity_val_file,  # 不用entity的时候注释掉
    train_vit_org_val=args.vit_org_val_file,  # vit
    train_vit_vg_val=args.vit_vg_val_file,  # vit
    # train_clip_org_val=args.clip_org_val_file,  # clip
    # train_clip_vg_val=args.clip_vg_val_file,  # clip
    test_path=args.test_file,
    test_pic_path=args.pic_test_file,
    test_head_ent_path=args.head_entity_test_file,  # 不用entity的时候注释掉
    test_tail_ent_path=args.tail_entity_test_file,  # 不用entity的时候注释掉
    train_vit_org_test=args.vit_org_test_file,  # vit
    train_vit_vg_test=args.vit_vg_test_file,  # vit
    # train_clip_org_test=args.clip_org_test_file,  # clip
    # train_clip_vg_test=args.clip_vg_test_file,  # clip
    model=model,
    ckpt=ckpt_path,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw',
    sample_ratio=args.sample_ratio
)

# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Val
for loader in [framework.val_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Val set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))

# Test
for loader in [framework.test_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Test set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))


    # MI可视化
    # print("result::",result)
    # print("correct_category::", correct_category)
    # print("org_category::", org_category)
    # print("n_category::", n_category)
    # print("data_pred_t::", data_pred_t)
    # print("data_pred_f::", data_pred_f)
    # print("id_list::", id_list)
    # print("feature_list::", feature_list)
    # workbook = openpyxl.Workbook()
    # sheet = workbook.active
    # file = "/home/zhangweiqi/TMR-main/ourResult/base_caption_mi/result.xlsx"
    # # epoch_name = epoch
    # # file_name = file.format(epoch_name)
    # for i, value in enumerate(data_pred_t):
    #     sheet.cell(row=i + 1, column=1, value=value)
    #
    # for i, value in enumerate(data_pred_f):
    #     sheet.cell(row=i + 1, column=2, value=value)
    #
    # for i, value in enumerate(id_list):
    #     sheet.cell(row=i + 1, column=3, value=value)
    #
    # workbook.save(file)

    # with open('./results/test/' + args.ckpt + '_result3123.txt', 'w') as f:
    #     for i in range(len(rel2id)):
    #         f.write(str(id2rel[i]) + ':' + str(acc_category[i]) + 'the count is ' + str(n_category[i]) + '\n')
    #     f.write('Test set results: \n')
    #     f.write('Accuracy: {}\n'.format(result['acc']))
    #     f.write('Micro precision: {}\n'.format(result['micro_p']))
    #     f.write('Micro recall: {}\n'.format(result['micro_r']))
    #     f.write('Micro F1: {}\n'.format(result['micro_f1']))
    # with open('./results/test/' + args.ckpt + '_data_with_pred_t123124.json', 'w', encoding='UTF-8') as f1:
    #     for i in range(len(data_pred_t)):
    #         f1.write(data_pred_t[i] + "\n")
    # with open('./results/test/' + args.ckpt + '_data_with_pred_f412312.json', 'w', encoding='UTF-8') as f2:
    #     for i in range(len(data_pred_f)):
    #         f2.write(data_pred_f[i] + "\n")
