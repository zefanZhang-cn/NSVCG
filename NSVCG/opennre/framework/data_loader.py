import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import timm
import cv2
from tqdm import tqdm, trange
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, DebertaTokenizer, RobertaTokenizer
import pdb
import h5py


class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    # def __init__(self, text_path, pic_path, head_ent_path, tail_ent_path, clip_org_path, clip_vg_path, rel2id,
    def __init__(self, text_path, pic_path, head_ent_path, tail_ent_path, vit_org_path, vit_vg_path, rel2id,
                 tokenizer, sample_ratio, kwargs):
        """
        Args:
            text_path: path of the input text
            pic_path: path of the input image
            head_ent_path or tail_ent_path: path of the input entity caption
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            sample_ratio: training data sampling ratio
        """

        super().__init__()
        self.text_path = text_path
        self.head_ent_path = head_ent_path  # entity caption路径
        self.tail_ent_path = tail_ent_path  # entity caption路径
        self.vit_org_path = vit_org_path
        self.vit_vg_path = vit_vg_path
        # self.clip_org_path = clip_org_path
        # self.clip_vg_path = clip_vg_path
        if 'train' in text_path:
            mode = 'train'
            self.mode = 'train'
        elif 'val' in text_path:
            mode = 'val'
            self.mode = 'val'
        else:
            mode = 'test'
            self.mode = 'test'

        # get generated images path

        # self.pic_path_FineGrained_dif = pic_path.replace('img_org', 'diffusion_pic')
        # #get original images path
        # self.pic_path_FineGrained_ori = pic_path
        # self.pic_path_CoarseGrained_ori = pic_path.replace('org', 'vg')

        # Load the text file
        self.data = []
        f = open(text_path, encoding='UTF-8')
        f_lines = f.readlines()
        for i1 in tqdm(range(len(f_lines))):
            line = f_lines[i1].rstrip()  # 这就是文本数据的一行，也就是一条文本数据
            if len(line) > 0:
                dic1 = eval(line)
                self.data.append(dic1)
        f.close()

        self.rel2id = rel2id
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))
        # Load the head entity file
        self.head_ent_data = []
        f_head = open(head_ent_path, encoding='UTF-8')
        f_lines_head = f_head.readlines()
        for i3 in tqdm(range(len(f_lines_head))):
            line = f_lines_head[i3].rstrip()  # caption的一条数据，由img_id和text组成
            if len(line) > 0:
                dic2 = eval(line)
                self.head_ent_data.append(dic2)
        f_head.close()

        logging.info(
            "Loaded head entity dataset {} with {} lines.".format(head_ent_path, len(self.head_ent_data)))

        # Load the tail entity file
        self.tail_ent_data = []
        f_tail = open(tail_ent_path, encoding='UTF-8')
        f_lines_tail = f_tail.readlines()
        for i3 in tqdm(range(len(f_lines_tail))):
            line = f_lines_tail[i3].rstrip()  # caption的一条数据，由img_id和text组成
            if len(line) > 0:
                dic2 = eval(line)
                self.tail_ent_data.append(dic2)
        f_tail.close() 

        logging.info(
            "Loaded tail entity dataset {} with {} lines.".format(tail_ent_path, len(self.tail_ent_data)))

        # Load ViT org file
        self.vit_org_file = h5py.File(self.vit_org_path, 'r')
        logging.info("Loaded ViT org dataset")

        # Load ViT vg file
        self.vit_vg_file = h5py.File(self.vit_vg_path, 'r')
        logging.info("Loaded ViT vg dataset")

        # if mode=="train":
        #     self.blip_feat=h5py.File("/home/zhangweiqi/test/compare_direct_describe_image_new_features_train.h5", 'r')
        #     self.obj_blip_feat=h5py.File("/home/zhangweiqi/test/obj_features_train.h5", 'r')
        # elif mode=="test":
        #     self.blip_feat = h5py.File("/home/zhangweiqi/test/compare_direct_describe_image_new_features_test.h5", 'r')
        #     self.obj_blip_feat = h5py.File("/home/zhangweiqi/test/obj_features_test.h5", 'r')
        # else:
        #     self.blip_feat = h5py.File("/home/zhangweiqi/test/compare_direct_describe_image_new_features_val.h5", 'r')
        #     self.obj_blip_feat = h5py.File("/home/zhangweiqi/test/obj_features_val.h5", 'r')


        if mode=="train":
            self.blip_feat=h5py.File("/home/zhangweiqi/test/new_features_train.h5", 'r')#/home/zhangweiqi/test/new_features_train.h5  /mnt/ssd/zhangweiqi/vgmre/rebuttal_for_prompt/parase5_part_features_train.h5
            self.obj_blip_feat=h5py.File("/home/zhangweiqi/test/obj_features_train.h5", 'r')
        elif mode=="test":
            self.blip_feat = h5py.File("/home/zhangweiqi/test/new_features_test.h5", 'r')
            self.obj_blip_feat = h5py.File("/home/zhangweiqi/test/obj_features_test.h5", 'r')
            self.ground_obj_feat = h5py.File("/home/zhangweiqi/test/ground/ground_obj_features.h5", 'r')
            self.label_ground_obj_feat = h5py.File("/home/zhangweiqi/test/ground/label_ground_obj_features.h5", 'r')
            with open('/home/zhangweiqi/test/ground/data_to_img.json', 'r') as f:
                self.data_to_img = json.load(f)
        else:
            self.blip_feat = h5py.File("/home/zhangweiqi/test/new_features_val.h5", 'r')
            self.obj_blip_feat = h5py.File("/home/zhangweiqi/test/obj_features_val.h5", 'r')



        # # Load clip org file
        # self.clip_org_file = h5py.File(self.clip_org_path, 'r')
        # logging.info("Loaded CLIP org dataset")
        #
        # # Load clip vg file
        # self.clip_vg_file = h5py.File(self.clip_vg_path, 'r')
        # logging.info("Loaded CLIP vg dataset")

        # get the path of reflecting dict
        self.img_aux_path_dif = text_path.replace('ours_{}.txt'.format(mode), 'mre_dif_{}_dict.pth'.format(mode))
        self.img_aux_path_ori = text_path.replace('ours_{}.txt'.format(mode), 'mre_{}_dict.pth'.format(mode))
        # load the reflecting dict
        self.state_dict_dif = torch.load(self.img_aux_path_dif)
        self.state_dict_ori = torch.load(self.img_aux_path_ori)

        # get the path of correlation scores
        self.weak_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_weak.txt'.format(mode))
        self.strong_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_strong.txt'.format(mode))
        self.weak_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_weak.txt'.format(mode))
        self.strong_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_strong.txt'.format(mode))
        # load the correlation scores
        with open(self.weak_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_ori = {}
            for line in lines:
                # eg：
                # img_id_key = twitter_stream_2018_10_10_9_0_2_192.jpg,
                # score = 0.998742938041687
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_ori[img_id_key] = score
        with open(self.strong_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_ori = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_ori[img_id_key] = score
        with open(self.weak_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_dif[img_id_key] = score
        with open(self.strong_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_dif[img_id_key] = score

        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.tokenizer_cap = BertTokenizer.from_pretrained('/home/zhangweiqi/.cache/huggingface/hub/bert-base-uncased')

        # get the path of phrases 短语
        self.phrase_path = text_path.replace('ours_{}.txt'.format(mode), 'phrase_text_{}.json'.format(mode))
        f_grounding = open(self.phrase_path, 'r')
        self.phrase_data = json.load(f_grounding)

        self.sample_ratio = sample_ratio
        sample_indexes = random.choices(list(range(len(f_lines))), k=int(len(f_lines) * sample_ratio))
        if 'train' in pic_path:
            new_state_dict = {}
            new_state_dict2 = {}
            new_grounding_text = {}
            num = 0
            for idx in sample_indexes:
                new_state_dict[num] = self.state_dict_dif[idx]
                new_state_dict2[num] = self.state_dict_ori[idx]
                new_grounding_text[str(num)] = self.phrase_data[str(idx)]
                num += 1
            self.state_dict_dif = new_state_dict
            self.state_dict_ori = new_state_dict2
            self.phrase_data = new_grounding_text
            tmp_data = [self.data[idx] for idx in sample_indexes]
            self.data = tmp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aux_imgs_ori = []
        for i in range(3):
            # VIT get visual objects from original images
            if len(self.state_dict_ori[index]) > i:
                elements = list(self.vit_vg_file.keys())
                imgid = self.state_dict_ori[index][i]
                if imgid in elements:
                    dataset = self.vit_vg_file[imgid]
                    aux_img_ori = dataset[:]
                    # image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
                    # img_features_ori = self.transform(image_ori_object).tolist()
                    # aux_img_ori = img_features_ori
                    aux_imgs_ori.append(torch.tensor(aux_img_ori).squeeze(0).tolist())

            # # CLIP get visual objects from original images
            # if len(self.state_dict_ori[index]) > i:
            #     elements = list(self.clip_vg_file.keys())
            #     imgid = self.state_dict_ori[index][i]
            #     if imgid in elements:
            #         dataset = self.clip_vg_file[imgid]
            #         # aux_img_ori = dataset[:] CLIP版本取[3]，其余不取
            #         aux_img_ori = dataset[:][3]
            #         # image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
            #         # img_features_ori = self.transform(image_ori_object).tolist()
            #         # aux_img_ori = img_features_ori
            #         aux_imgs_ori.append(torch.tensor(aux_img_ori).squeeze(0).tolist())

        for i in range(3 - len(aux_imgs_ori)):
            # aux_imgs_ori.append(torch.zeros((3, 224, 224)).tolist()
            aux_imgs_ori.append(torch.zeros((197, 768)).tolist())  # vit
            # aux_imgs_ori.append(torch.zeros((257, 1024)).tolist())  # clip

        item = self.data[index]
        item['grounding'] = self.phrase_data[str(index)]  # phrase


        #item["token"]=item["token"]+["please identify the relation between "+item['h']["name"]+" and "+item['t']["name"]]


        seq = list(self.tokenizer(item, **self.kwargs))  # input ids
        img_id = item['img_id']  # img_id
        h = item['h']  # head entity
        t = item['t']  # tail entity



        blip_obj_feat=torch.tensor(self.obj_blip_feat[img_id][:]).squeeze(1)
        aaaa,bbbb=blip_obj_feat.size()


        img_patch_len=100#100
        object_num=12     #10


        if aaaa>object_num:
            blip_obj_feat=blip_obj_feat[:object_num,:]
        elif aaaa<object_num:
            pad=torch.zeros(object_num-aaaa,4096)
            blip_obj_feat=torch.cat([blip_obj_feat,pad],dim=0)

        blip_feat=torch.tensor(self.blip_feat[(h["name"]+t["name"]+img_id).replace(" ","")][:]).squeeze(0)

        aaaa,bbbb=blip_feat.size()
        if aaaa>img_patch_len:
            blip_feat=blip_feat[:img_patch_len,:]
        elif aaaa<img_patch_len:
            pad=torch.zeros(img_patch_len-aaaa,4096)
            blip_feat=torch.cat([blip_feat,pad],dim=0)
        blip_feat=blip_feat.unsqueeze(0)
        blip_obj_feat = blip_obj_feat.unsqueeze(0)
        # print(blip_obj_feat.shape)
        # print(fsfsaf)





        # 获取head entity的id和text
        for i, line in enumerate(self.head_ent_data):
            if img_id == line['img_id']:
                head_ent_id = line['img_id']
                head_ent_text = line['text']
        #head_ent_text = h["name"] + ": " + head_ent_text.split(".")[0]
        head_ent_text = item['grounding']+ ": " + head_ent_text.split(".")[0]
        # 对entity的text进行tokenizer编码
        head_encode_entity = self.tokenizer_cap.encode_plus(text=head_ent_text, max_length=70, truncation=True,
                                                            padding='max_length')
        head_entity_input_ids, head_entity_token_type_ids, head_entity_attention_mask = head_encode_entity['input_ids'], \
                                                                                        head_encode_entity[
                                                                                            'token_type_ids'], \
                                                                                        head_encode_entity[
                                                                                            'attention_mask']
        head_entity_input_ids, head_entity_token_type_ids, head_entity_attention_mask = torch.tensor(
            head_entity_input_ids).unsqueeze(0), torch.tensor(head_entity_token_type_ids).unsqueeze(0), torch.tensor(
            head_entity_attention_mask).unsqueeze(0)



        # 获取tail entity的id和text
        for i, line in enumerate(self.tail_ent_data):
            if img_id == line['img_id']:
                tail_ent_id = line['img_id']
                tail_ent_text = line['text']

        #tail_ent_text = t["name"]+": "+tail_ent_text.split(".")[0]
        tail_ent_text = item['grounding'] + ": " + tail_ent_text.split(".")[0]
        # 对entity的text进行tokennizer编码
        tail_encode_entity = self.tokenizer_cap.encode_plus(text=tail_ent_text, max_length=70, truncation=True,
                                                            padding='max_length')
        tail_entity_input_ids, tail_entity_token_type_ids, tail_entity_attention_mask = tail_encode_entity['input_ids'], \
                                                                                        tail_encode_entity[
                                                                                            'token_type_ids'], \
                                                                                        tail_encode_entity[
                                                                                            'attention_mask']
        tail_entity_input_ids, tail_entity_token_type_ids, tail_entity_attention_mask = torch.tensor(
            tail_entity_input_ids).unsqueeze(0), torch.tensor(tail_entity_token_type_ids).unsqueeze(0), torch.tensor(
            tail_entity_attention_mask).unsqueeze(0)





        # ViT load the original image
        org_elements = list(self.vit_org_file.keys())
        if img_id in org_elements:
            dataset_org = self.vit_org_file[img_id]
            img_ori = dataset_org[:]
            img_ori = torch.tensor(img_ori).squeeze(0).to(torch.float32).tolist()


        # # CLIP load the original image
        # org_elements = list(self.clip_org_file.keys())
        # if img_id in org_elements:
        #     dataset_org = self.clip_org_file[img_id]
        #     # img_ori = dataset_org[:] CLIP版本取[3]，其余不取
        #     img_ori = dataset_org[:][3]
        #
        #     img_ori = torch.tensor(img_ori).squeeze(0).to(torch.float32).tolist()
        #     # clip: img_ori是1,257,1024

        pic_ori = [img_ori]
        pic_ori_objects = aux_imgs_ori

        np_pic_ori = np.array(pic_ori).astype(np.float32)
        np_pic_ori_objects = np.array(pic_ori_objects).astype(np.float32)
        weight = [self.weak_ori[img_id], self.strong_ori[img_id]]

        list_pic_ori = list(torch.tensor(np_pic_ori).unsqueeze(0))
        list_pic_ori_objects = list(torch.tensor(np_pic_ori_objects).unsqueeze(0))








        #resnet yong
        # aux_imgs_ori = []
        #
        # for i in range(3):
        #     # get visual objects from generated images
        #
        #     # get visual objects from original images
        #     if len(self.state_dict_ori[index])>i:
        #         image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
        #         img_features_ori = self.transform(image_ori_object).tolist()
        #         aux_img_ori = img_features_ori
        #         aux_imgs_ori.append(aux_img_ori)
        #
        #
        # for i in range(3 - len(aux_imgs_ori)):
        #     aux_imgs_ori.append(torch.zeros((3, 224, 224)).tolist())
        #
        # image_ori = cv2.imread((os.path.join(self.pic_path_FineGrained_ori, self.data[index]['img_id'])))
        # size = (224, 224)
        # img_features_ori = cv2.resize(image_ori, size, interpolation=cv2.INTER_AREA)
        # img_features_ori = torch.tensor(img_features_ori)
        # img_features_ori = img_features_ori.transpose(1, 2).transpose(0, 1)
        # img_ori = torch.reshape(img_features_ori, (3, 224, 224)).to(torch.float32).tolist()
        # pic_ori_objects = aux_imgs_ori
        # pic_ori = [img_ori]
        # np_pic1 = np.array(pic_ori).astype(np.float32)
        # np_pic2 = np.array(pic_ori_objects).astype(np.float32)
        #
        # list_pic_ori = list(torch.tensor(np_pic1).unsqueeze(0))
        # list_pic_ori_objects = list(torch.tensor(np_pic2).unsqueeze(0))
        
        
        
        
        
        
        #for ground test
        if self.mode=="test":
            ground_obj=torch.tensor(np.array(self.ground_obj_feat[self.data_to_img[h["name"]+t["name"]+img_id]])).squeeze(1)
            label_ground_obj=list(self.label_ground_obj_feat[self.data_to_img[h["name"]+t["name"]+img_id]])
            final_ground_obj=[]
            final_label=[]
            false=[]
            for i in range(0,ground_obj.size(0)):
                if label_ground_obj[i]==1:
                    final_ground_obj.append(ground_obj[i])
                    final_label.append(1)
                else:
                    false.append(ground_obj[i])
            t=0
            while len(final_ground_obj)<4 and len(false)>t:
                final_ground_obj.append(false[t])
                final_label.append(0)
                t=t+1
            while len(final_ground_obj) < 4:
                pad = torch.zeros( 4096)
                final_ground_obj.append(pad)
                final_label.append(0)
            final_label=torch.tensor(final_label)[:4].unsqueeze(0)
            final_ground_obj=torch.stack(final_ground_obj,dim=0)[:4,:].unsqueeze(0)


        else:
            final_label=torch.tensor([0])
            final_ground_obj=torch.tensor([0])



















        # res = [self.rel2id[item['relation']]] + [img_id] + seq + list_pic_dif + list_pic_ori + list_pic_dif_objects + list_pic_ori_objects + [torch.tensor(weight).reshape(1,4)]+[caption_input_ids]+[caption_token_type_ids]+[caption_attention_mask]+[head_entity_input_ids]+[head_entity_token_type_ids]+[head_entity_attention_mask]+[tail_entity_input_ids]+[tail_entity_token_type_ids]+[tail_entity_attention_mask]

        res = [self.rel2id[item['relation']]] + [img_id] + [index] + seq + list_pic_ori + list_pic_ori_objects + [
            torch.tensor(weight).reshape(1, 2)] + [head_entity_input_ids] + [head_entity_token_type_ids] + [
                  head_entity_attention_mask] + [tail_entity_input_ids] + [tail_entity_token_type_ids] + [
                  tail_entity_attention_mask]+[blip_feat]+[blip_obj_feat]+[final_ground_obj]+[final_label]

        return res  # label, seq1, seq2, ...,pic,caption的传入bert的三个参数

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        img_id = data[1]
        index = data[2]
        seqs = data[3:]  #  这里改了
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            # print(seq)
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + [img_id] +[index] + batch_seqs

    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        correct_category = np.zeros([31, 1])
        org_category = np.zeros([31, 1])
        n_category = np.zeros([31, 1])
        data_with_pred_T = []
        data_with_pred_F = []
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        y_pred = []
        y_gt = []
        for i in range(total):
            y_pred.append(pred_result[i])
            y_gt.append(self.rel2id[self.data[i]['relation']])
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]  # Ground Truth Label
                n_category[golden] += 1
            data_with_pred = (str(self.data[i]) + str(pred_result[i]))
            if golden == pred_result[i]:
                correct += 1
                data_with_pred_T.append(data_with_pred)
                if golden != neg:
                    correct_positive += 1
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1
            else:
                data_with_pred_F.append(data_with_pred)
            if golden != neg:
                gold_positive += 1
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result, correct_category, org_category, n_category, data_with_pred_T, data_with_pred_F


def SentenceRELoader(text_path, pic_path, head_ent_path, tail_ent_path, vit_org_path, vit_vg_path, rel2id,
                     tokenizer,
                     batch_size, shuffle, num_workers=8, sample_ratio=1.0, collate_fn=SentenceREDataset.collate_fn,
                     **kwargs):
    dataset = SentenceREDataset(text_path=text_path, pic_path=pic_path, head_ent_path=head_ent_path,
                                tail_ent_path=tail_ent_path,
                                vit_org_path=vit_org_path,
                                vit_vg_path=vit_vg_path,
                                # clip_org_path=clip_org_path,
                                # clip_vg_path=clip_vg_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                sample_ratio=sample_ratio,
                                kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    # 由dataloader传出去的就是之前的train_data之类的，然后他们是由多个batch组成，所以我们把caption_sentence那两个从这里传出去
    return data_loader




