import pickle
import json
import torch
import numpy as np
import pandas as pd
import glob
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
from TorchSSL.GAT_utils import accuracy
import contextlib
from TorchSSL.train_utils import AverageMeter

from .flexmatch_utils import consistency_loss, Get_Scalar
from TorchSSL.train_utils import ce_loss, wd_loss, EMA, Bn_Controller

from sklearn.metrics import *
from copy import deepcopy


class FlexMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Flexmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FlexMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, idx_train, idx_val, idx_test, idx_train_unlb, features, unlb_features, adj, labels):
        self.idx_train = idx_train.cuda()
        self.idx_val = idx_val.cuda()
        self.idx_test = idx_test.cuda()
        self.idx_train_unlb = idx_train_unlb.cuda()
        self.features = features.cuda()
        self.unlb_features = unlb_features.cuda()
        self.adj = adj.cuda()
        self.labels = labels.cuda()

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def model_train(self, args, logger=None):

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        p_target = None
        p_model = None

        if args.resume:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.idx_train_unlb),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda()

        classwise_acc = torch.zeros((2,)).cuda()

        loss_values = []
        bad_counter = 0
        best = args.epochs + 1
        best_epoch = 0
        x_ulb_idx = torch.LongTensor(range(500)).cuda()
        for epoch in range(args.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.idx_train_unlb):  # not all(5w) -1
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            output = self.model(self.features, self.adj)
            output_ul = self.model(self.unlb_features, self.adj)
            sup_loss = ce_loss(output[self.idx_train], self.labels[self.idx_train], reduction='mean')
            # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # hyper-params for update
            T = self.t_fn(self.it)
            p_cutoff = self.p_fn(self.it)

            unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(output_ul[self.idx_train_unlb],
                                                                            output[self.idx_train_unlb],
                                                                            classwise_acc,
                                                                            p_target,
                                                                            p_model,
                                                                            'ce', T, p_cutoff,
                                                                            use_hard_labels=args.hard_label,
                                                                            use_DA=args.use_DA)

            if x_ulb_idx[select == 1].nelement() != 0:
                selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

            total_loss = sup_loss + self.lambda_u * unsup_loss

            # parameter updates
            total_loss.backward()
            self.optimizer.step()
            self.ema.update()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                self.model.eval()
                self.ema.apply_shadow()
                output = self.model(self.features, self.adj)

            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
            acc_val, recall_val, auc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
            self.ema.restore()
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(total_loss.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'recall_val: {:.4f}'.format(recall_val),
                  'auc_val: {:.4f}'.format(auc_val))

            loss_values.append(loss_val.data.item())

            torch.save(self.model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print('Loading {}th epoch'.format(best_epoch))
        self.model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        # Testing
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test, recall_test, auc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()),
              "recall= {:.4f}".format(recall_test),
              "auc= {:.4f}".format(auc_test))

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

if __name__ == "__main__":
    pass
