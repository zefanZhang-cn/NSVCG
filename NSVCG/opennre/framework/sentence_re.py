import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
import numpy as np
from sklearn import metrics
from seqeval.metrics import f1_score
import pdb
import torch.nn.functional as F
class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=23, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += (1 - alpha)
        self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class SentenceRE(nn.Module):
    def __init__(self,
                 model,
                 train_path,
                 train_pic_path,
                 train_head_ent_path,  # 不用entity的时候注释掉
                 train_tail_ent_path,  # 不用entity的时候注释掉
                 train_vit_org_train,
                 train_vit_vg_train,
                 # train_clip_org_train,
                 # train_clip_vg_train,
                 val_path,
                 val_pic_path,
                 val_head_ent_path,  # 不用entity的时候注释掉
                 val_tail_ent_path,  # 不用entity的时候注释掉
                 train_vit_org_val,
                 train_vit_vg_val,
                 # train_clip_org_val,
                 # train_clip_vg_val,
                 test_path,
                 test_pic_path,
                 test_head_ent_path,  # 不用entity的时候注释掉
                 test_tail_ent_path,  # 不用entity的时候注释掉
                 train_vit_org_test,
                 train_vit_vg_test,
                 # train_clip_org_test,
                 # train_clip_vg_test,
                 ckpt,
                 batch_size=64,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-3,
                 warmup_step=300,
                 opt='adamw',
                 sample_ratio=1.0):

        super().__init__()
        self.max_epoch = max_epoch
        self.testing = False
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                train_pic_path,
                train_head_ent_path,  # 不用entity的时候注释掉
                train_tail_ent_path,  # 不用entity的时候注释掉
                train_vit_org_train,
                train_vit_vg_train,
                # train_clip_org_train,
                # train_clip_vg_train,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                sample_ratio=sample_ratio)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                val_pic_path,
                val_head_ent_path,  # 不用entity的时候注释掉
                val_tail_ent_path,  # 不用entity的时候注释掉
                train_vit_org_val,
                train_vit_vg_val,
                # train_clip_org_val,
                # train_clip_vg_val,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                test_pic_path,
                test_head_ent_path,  # 不用entity的时候注释掉
                test_tail_ent_path,  # 不用entity的时候注释掉
                train_vit_org_test,
                train_vit_vg_test,
                # train_clip_org_test,
                # train_clip_vg_test,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss=focal_loss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def compute_kl_loss(self,p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    def criterion_KL(self,x, y):
        x_log = F.log_softmax(x, dim=1)
        # 只转化为概率
        y = F.softmax(y, dim=1)
        kl = nn.KLDivLoss(reduction='batchmean').cuda()
        out = kl(x_log, y)
        return out

    def criterion_KL2(self,x, y):
        x_log = F.log_softmax(x, dim=1)
        # 只转化为概率
        y = F.softmax(y, dim=1)
        kl = nn.KLDivLoss(reduction='batchmean').cuda()
        out = kl(x_log, y)
        return out


    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        loader = [self.train_loader]
        final=[]

        iii=0
        for epoch in range(self.max_epoch):
            self.train()
            iii+=1
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_f1 = AverageMeter()
            # head_index = {}
            # tail_index = {}
            # vision_index = {}
            for l1 in loader:
                t = tqdm(l1, ncols=110)
                for iter, data in enumerate(t):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass

                    label = data[0]
                    img_id = data[1]
                    index = data[2]
                    args = data[3:]
                    logits, rep,rep_old,c,all_right = self.parallel_model(*args, label)

                    if all_right!=None:
                        final=final+all_right

                    #kl_batch_loss = self.criterion_KL(rep, rep_old)
                    #loss = self.criterion(logits, label)+self.criterion_KL2(rep_old, F.one_hot(label, 23).float()) #+ kl_batch_loss
                    loss =2*self.criterion(c, label)+self.criterion(rep_old, label)#+self.compute_kl_loss(self.criterion(c, label),self.criterion(rep_old, label))#+self.criterion_KL2(rep_old, F.one_hot(label, 23).float())
                    #loss = 2 * self.focal_loss(c, label) + self.focal_loss(rep_old, label)
                    #if iii>3:
                     #   loss+=self.criterion_KL2(rep_old, F.one_hot(label, 23).float())
                    #loss = self.focal_loss(rep_old, label)
                    logits = rep_old
                    rep=rep_old
                    # for i in range(len(head_proto_index)):
                    #     head_index[head_proto_index[i].item()] = head_proto_index[i].item()
                    #
                    # for i in range(len(tail_proto_index)):
                    #     tail_index[tail_proto_index[i].item()] = tail_proto_index[i].item()
                    #
                    # for i in range(len(vision_proto_index)):
                    #     vision_index[vision_proto_index[i].item()] = vision_proto_index[i].item()

                    # logits2, rep = self.parallel_model(*args, label)
                    # loss2 = self.criterion(logits2, label)
                    #
                    # ce_batch_loss=(loss+loss2)/2
                    # kl_batch_loss=self.compute_kl_loss(loss, loss2)
                    # cl_loss=ce_batch_loss+4*kl_batch_loss
                    # loss=cl_loss

                    score, pred = logits.max(-1)  # (B)

                    acc = float((pred == label).long().sum()) / label.size(0)
                    f1 = metrics.f1_score(pred.cpu(), label.cpu(), average='macro')
                    # Log
                    avg_loss.update(loss.item(), 1)
                    avg_acc.update(acc, 1)
                    avg_f1.update(f1, 1)
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, f1=avg_f1.avg)
                    # Optimize
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            import time
            start = time.time()
            result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
                self.test_loader)
            end = time.time()
            print("all time: "+str(end-start))
                
                
                
            result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
                self.val_loader)
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                best_metric = result[metric]

                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                # logging.info("Best ckpt and saved.")Å
                logging.info("Best ckpt and saved.")
            # with open('./head_index.josn', 'w') as ff:
            #     json.dump(head_index, ff)
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        id_list = []
        feature_list = []
        final = {}
        head_index = {}
        tail_index = {}
        vision_index = {}
        ff=[]

        with torch.no_grad():
            t = tqdm(eval_loader, ncols=110)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                img_id = data[1]
                index = data[2]
                args = data[3:]
                logits, rep,rep_old,c,all_right = self.parallel_model(*args, label)
                if all_right!=None:
                    ff=ff+all_right

                #kl_batch_loss = self.criterion_KL(rep, loss2)
                #loss = self.criterion(logits, label)+self.criterion_KL2(rep_old, F.one_hot(label, 23).float()) #+ kl_batch_loss
                loss = self.criterion(rep_old, label)+2*self.criterion(c, label)
                #loss = 2 * self.focal_loss(c, label) + self.focal_loss(rep_old, label)
                #loss = self.focal_loss(rep_old, label)
                logits=rep_old
                rep=rep_old
                # logits2, rep = self.parallel_model(*args, label)
                # loss2 = self.criterion(logits2, label)
                #
                # ce_batch_loss = (loss + loss2) / 2
                # kl_batch_loss = self.compute_kl_loss(loss, loss2)
                # cl_loss = ce_batch_loss + 4 * kl_batch_loss
                # loss = cl_loss

                # for i in range(len(head_proto_index)):
                #     head_index[index[i]] = head_proto_index[i].item()
                #
                # for i in range(len(tail_proto_index)):
                #     tail_index[index[i]] = tail_proto_index[i].item()
                #
                # for i in range(len(vision_proto_index)):
                #     vision_index[index[i]] = vision_proto_index[i].item()

                # loss = self.criterion(logits, label)
                score, pred = logits.max(-1)  # (B)

                # for i in range(len(index)):
                #     if pred[i] == label[i]:
                #         final[index[i]] = 1
                #     else:
                #         final[index[i]] = 0

                id_list.append(img_id)
                feature_list.append(rep)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), pred.size(0))
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
        if ff!=[]:
            print('----------------------------------------')
            print("VGMRE_ACC")
            print(sum(ff)/len(ff))
            print('----------------------------------------')
        result, correct_category, org_category, n_category, data_pred_t, data_pred_f = eval_loader.dataset.eval(
            pred_result)

        # print(head_index)
        # with open('./head_index.josn', 'w') as ff1:
        #     json.dump(head_index, ff1)
        # # # print(tail_index)
        # with open('./tail_index.josn', 'w') as ff2:
        #     json.dump(tail_index, ff2)
        # # # print(vision_index)
        # with open('./vision_index.josn', 'w') as ff3:
        #     json.dump(vision_index, ff3)

        # save prediction into JSON
        with open('./pred_results.josn', 'w') as f2:
            json.dump(pred_result, f2)
        return result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

