import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import DebertaModel, RobertaModel
import transformers
from .modeling_bert import BertModel
import math
from torch.nn import functional as F
import json
from torch.nn.functional import gelu, relu, tanh
import numpy as np
from torch import nn
from torchvision.models import resnet50, resnet101, resnet152
import timm
import pdb
import random


class MyAttention(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.merge_fr = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ELU(inplace=True)

        )

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, y):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        # linear projection
        q = self.linear_q(x)  # abc
        k = self.linear_k(y)  # adc
        v = self.linear_v(y)  # adc

        attention = torch.bmm(q, k.transpose(-2, -1))  # abd
        scale = v.size(-1) ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        output = torch.bmm(attention, v)  # abc

        output = self.layer_norm(output)
        output = self.merge_fr(output)

        return output


class TMR_RE(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        max_length: max length of sentence
        pretrain_path: path of pretrain model
        blank_padding: need padding or not
        mask_entity: mask the entity tokens or not
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.hidden_size = 768 * 2

        self.nclass = 300
        # print(timm.create_model('resnet152').default_cfg)  #查看并下载resnet
        # self.model_resnet152 = timm.create_model('resnet152', pretrained=True,pretrained_cfg_overlay=dict(file='/home/zhangweiqi/TMR-main/resnet152_a1h-dc400468.pth')) # get the pre-trained ResNet model for the image
        # self.model_resnet101 = timm.create_model('resnet101', pretrained=True,pretrained_cfg_overlay=dict(file='/home/zhangweiqi/TMR-main/resnet101_a1h-36d3f2aa.pth')) # get the pre-trained ResNet model for the image
        #resnet yong
        # self.model_resnet50 = timm.create_model('resnet50', pretrained=True, pretrained_cfg_overlay=dict(
        #     file='/home/zhangweiqi/TMR-main/resnet50_a1_0-14fe96d1.pth'))  # get the pre-trained ResNet model for the image
        # for param in self.model_resnet50.parameters():
        #     param.requires_grad = True
        # self.linear_pic = nn.Linear(2048, self.hidden_size // 2)



        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(
            '/home/zhangweiqi/.cache/huggingface/hub/bert-base-uncased')  # get the pre-trained BERT model for the text
        self.bert2 = transformers.BertModel.from_pretrained('/home/zhangweiqi/.cache/huggingface/hub/bert-base-uncased')

        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        # self.linear_final_woDif = nn.Linear(self.hidden_size * 2 + self.hidden_size // 2, self.hidden_size)

        self.linear_final_woDif2 = nn.Linear(self.hidden_size, self.hidden_size)
        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        self.linear_weights = nn.Linear(self.hidden_size * 3, 3)
        self.linear_phrases = nn.Linear(self.hidden_size // 2, self.hidden_size//2)
        # self.linear_extend_pic = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_extend_pic_old = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_extend_pic_new = nn.Linear(self.hidden_size // 2, self.hidden_size//2)
        #self.linear_extend_pic_new2 = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.dropout_linear = nn.Dropout(0.5)#0.5
        self.mlp_blip = nn.Sequential(
           #  nn.Linear(4096, 2048),
           # nn.ELU(),
           #  nn.LayerNorm(2048),
           #  nn.Dropout(0.5),
           #  nn.Linear(2048, 768)
            nn.Linear(4096, 768)
        )
        self.mlp_blip_obj = nn.Sequential(
           #  nn.Linear(4096, 2048),
           # nn.ELU(),
           #  nn.LayerNorm(2048),
           # nn.Dropout(0.5),
           #  nn.Linear(2048, 768)
            nn.Linear(4096, 768)
        )

        self.linear_final_woDif1 = nn.Linear(self.hidden_size * 2 + self.hidden_size // 2 + self.hidden_size,
                                             self.hidden_size)
        self.ll=nn.Linear(768*4,768)
        #self.ll2 = nn.Linear(768 * 4, 768)
        #resnetyong



    def forward(self,
                token,  # 先不用
                att_mask,  # 先不用
                pos1,
                pos2,
                token_phrase,  # 先不用
                att_mask_phrase,  # 先不用
                # image_diff,
                image_ori,
                # image_diff_objects,
                image_ori_objects,
                weights,
                head_entity_input_ids,
                head_entity_token_type_ids,
                head_entity_attention_mask,
                tail_entity_input_ids,
                tail_entity_token_type_ids,
                tail_entity_attention_mask,
                blip_feat,
                blip_obj_feat,
                ground,
                ground_label,
                label,
                ):



        #blip feat bs 90 4096
        #blip obj feat  bs 10 4096

        # vit原版
        # pic_ori = image_ori
        # pic_ori_ = torch.mean(pic_ori, dim=1)
        # # ViT原版pic_ori_objects
        # pic_ori_objects = torch.reshape(image_ori_objects, (-1, 3, 197, 768))
        # pic_ori_objects = torch.mean(pic_ori_objects, dim=2)
        # pic_ori_objects_ = torch.mean(pic_ori_objects, dim=1)
        
        



        pic_ori = self.mlp_blip(blip_feat)
        pic_ori_ = torch.mean(pic_ori, dim=1)
        # ViT原版pic_ori_objects
        pic_ori_objects = self.mlp_blip_obj(blip_obj_feat)
        pic_ori_objects_ = torch.mean(pic_ori_objects, dim=1)
        
        
        #ground test
        if ground_label.size()!=ground.size():
            ground=self.mlp_blip_obj(ground)



        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]  # bsz,128,768



        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]  # bsz,16,768

        hidden_k_text = self.linear_k_fine(pic_ori)
        hidden_v_text = self.linear_v_fine(pic_ori)
        pic_q_origin = self.linear_q_fine(hidden_text)  # 原版
        a,b=self.att1(pic_q_origin, hidden_k_text, hidden_v_text)
        pic_original = torch.sum(torch.tanh(a), dim=1)  # bsz,768


        hidden_k_phrases = self.linear_k_coarse(pic_ori_objects)
        hidden_v_phrases = self.linear_v_coarse(pic_ori_objects)
        pic_q_ori_objects = self.linear_q_coarse(hidden_phrases)  # 原版
        a1,b1=self.att1(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)
        pic_original_objects = torch.sum(torch.tanh(a1),
                                         dim=1)



        # coarse-grained textual features
        hidden_phrases = torch.sum(hidden_phrases, dim=1)
        hidden_phrases = self.linear_phrases(hidden_phrases)  # 32,2*768

        # Get entity start hidden state
        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)  # (B, H)
        # fine-grained textual features
        x = torch.cat([head_hidden, tail_hidden], dim=-1)  # bsz,2*768



        pic_ori_old = torch.tanh(self.linear_extend_pic_old(pic_original + pic_ori_)* weights[:, 1].reshape(-1, 1))  # 32,768
        pic_ori_new = torch.tanh(self.linear_extend_pic_new(pic_original_objects + pic_ori_objects_)* weights[:, 0].reshape(-1, 1))  # 32,768


        #noisy=self.ll2(torch.cat([x,torch.sum(torch.tanh(b), dim=1),torch.sum(torch.tanh(b1), dim=1)],dim=-1))
        noisy=None

        xx = torch.cat([x, hidden_phrases, pic_ori_new + pic_ori_old], dim=1)
        xx=self.ll(self.dropout_linear(xx)).unsqueeze(1)

        return xx.mean(1), xx, label,noisy,ground,ground_label,xx.mean(1),xx.mean(1)#pic_ori_old+pic_ori_new,pic_ori_old+pic_ori_new


    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        if len(item) == 5 and not isinstance(item, dict):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

            return indexed_tokens

        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        elif 'token' in item:
            sentence = item['token']
            is_token = True

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        phrases = item['grounding']
        phrases_token = self.tokenizer.encode_plus(text=phrases, max_length=16, truncation=True,
                                                            padding='max_length',add_special_tokens=False)

        token_phrases, att_mask_phrases = phrases_token['input_ids'], phrases_token['attention_mask']
        token_phrases, att_mask_phrases = torch.tensor(token_phrases).unsqueeze(0), torch.tensor(att_mask_phrases).unsqueeze(0)
        # token_phrases = self.tokenizer.convert_tokens_to_ids(phrases.split(' '))
        # while len(token_phrases) < 6:
        #     token_phrases.append(0)
        # token_phrases = token_phrases[:6]
        # token_phrases = torch.tensor(token_phrases).long().unsqueeze(0)
        # att_mask_phrases = torch.zeros(token_phrases.size()).long()
        return indexed_tokens, att_mask, pos1, pos2, token_phrases, att_mask_phrases

    # the attention mechanism
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def att1(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value),torch.matmul(1.0-att_map, value)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                # x = gelu(x)
                x = relu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x




