# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import os
import time
import shutil

from tqdm import tqdm
from dml.utils import *
from torch_model.model import Bind



class Ind_Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.model_name = config.save_name


        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_acc = 0

        self.model = Bind(self.num_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay, nesterov=self.nesterov)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=self.gamma, last_epoch=-1)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=self.gamma, last_epoch=-1)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate 5cv_ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=True)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        best_epoch = 0
        train_loss_list = []
        train_acc_list = []
        valid_loss_list = []
        valid_acc_list = []
        targets = ['non', 'rRNA', 'tRNA', 'snRNA', 'mRNA', 'SRP']
        valid_auc_list = dict()
        train_auc_list = dict()
        for target in targets:
            train_auc_list[target] = []
            valid_auc_list[target] = []

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]['lr'], )
            )

            # train for 1 epoch
            train_losses, train_accs, train_aucs = self.train_one_epoch(epoch)
            train_loss_list.append(train_losses.avg)
            train_acc_list.append(train_accs.avg)

            # evaluate on validation set
            valid_losses, valid_accs, valid_aucs = self.validate(epoch)
            valid_loss_list.append(valid_losses.avg)
            valid_acc_list.append(valid_accs.avg)

            self.scheduler.step()

            for i in range(len(targets)):
                print('partner: %s  -train_auc: %s  -valid_auc: %s' % (targets[i],
                                                                       str(round(train_aucs[targets[i]], 4)),
                                                                       str(round(valid_aucs[targets[i]], 4))))
                train_auc_list[targets[i]].append(train_aucs[targets[i]])
                valid_auc_list[targets[i]].append(valid_aucs[targets[i]])

            is_best = valid_accs.avg > self.best_valid_acc
            msg1 = "model: train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                best_epoch = epoch
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg))

            self.best_valid_acc = max(valid_accs.avg, self.best_valid_acc)
            self.save_checkpoint({'epoch': epoch + 1,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  'best_valid_acc': self.best_valid_acc,
                                  }, is_best)
            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                plot_loss_acc(epoch+1, train_loss=train_loss_list, train_acc=train_acc_list,
                              valid_loss=valid_loss_list, valid_acc=valid_acc_list, title='Loss&ACC',
                              save_path=self.logs_dir+'/train_loss_acc_'+self.model_name+'.jpg')
                plot_aucs(epoch+1, targets, train_auc_list, valid_auc_list, title='AUC',
                          save_path=self.logs_dir+'/train_auc_'+self.model_name+'.jpg')
                for target in targets:
                    print(
                        'partner: {}, valid_auc(best/recent): {:.3f}/{:.3f}'.format(target,
                                                                                 valid_auc_list[target][best_epoch],
                                                                                 valid_auc_list[target][epoch])
                    )
                return

        plot_loss_acc(self.epochs, train_loss=train_loss_list, train_acc=train_acc_list,
                      valid_loss=valid_loss_list, valid_acc=valid_acc_list, title='Loss&ACC',
                      save_path=self.logs_dir+'/train_loss_acc_'+self.model_name+'.jpg')
        plot_aucs(self.epochs, targets, train_auc_list, valid_auc_list, title='AUC',
                  save_path=self.logs_dir+'/train_auc_'+self.model_name+'.jpg')

        for target in targets:
            print(
                'partner: {}, valid_auc(best/recent): {:.3f}/{:.3f}'.format(target, valid_auc_list[target][best_epoch],
                                                                            valid_auc_list[target][epoch])
            )



    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        loss = AverageMeter()
        acc = AverageMeter()

        train_pred = []
        train_true = []

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            # each batch
            for i, (x, neibor_x, label) in enumerate(self.train_loader):
                if self.use_gpu:
                    x, neibor_x, label = x.cuda(), neibor_x.cuda(), label.cuda()
                x, neibor_x, label = Variable(x), Variable(neibor_x), Variable(label)
                train_true.extend(label.data.numpy())
                # initial gradients
                self.optimizer.zero_grad()

                # forward pass
                output = self.model(x, neibor_x)

                train_pred.extend(output.detach().numpy())

                ce_loss = self.loss_ce(output, label)
                loss.update(ce_loss.item(), x.size()[0])
                acc.update(accuracy(output.data, label.data), x.size()[0])
                ce_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model_loss: {:.3f} - model_acc: {:.3f}".format(
                            (toc - tic), loss.avg, acc.avg
                        )
                    )
                )
                self.batch_size = label.shape[0]
                pbar.update(self.batch_size)

        train_true = np.array(train_true)
        train_pred = np.array(train_pred)
        train_true = label_binarize(train_true, classes=[i for i in range(6)])
        roc_auc_model = dict()
        target = ['non', 'rRNA', 'tRNA', 'snRNA', 'mRNA', 'SRP']
        for i in range(len(target)):
            roc_auc_model[target[i]] = roc_auc_score(train_true[:, i], train_pred[:, i])
            # print('partner: %s  -train_auc: %s' % (target[i], str(round(roc_auc_model[target[i]], 4))))
        return loss, acc, roc_auc_model

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        loss = AverageMeter()
        acc = AverageMeter()

        valid_pred = []
        valid_true = []

        for i, (x, neibor_x, label) in enumerate(self.valid_loader):
            if self.use_gpu:
                x, neibor_x, label = x.cuda(), neibor_x.cuda(), label.cuda()
            x, neibor_x, label = Variable(x), Variable(neibor_x), Variable(label)

            valid_true.extend(label.data.numpy())

            # forward pass
            output = self.model(x, neibor_x)

            valid_pred.extend(output.detach().numpy())

            ce_loss = self.loss_ce(output, label)
            loss.update(ce_loss.item(), x.size()[0])
            acc.update(accuracy(output.data, label.data), x.size()[0])

        valid_true = np.array(valid_true)
        valid_pred = np.array(valid_pred)
        valid_true = label_binarize(valid_true, classes=[i for i in range(6)])
        roc_auc_model = dict()
        target = ['non', 'rRNA', 'tRNA', 'snRNA', 'mRNA', 'SRP']
        for i in range(len(target)):
            roc_auc_model[target[i]] = roc_auc_score(valid_true[:, i], valid_pred[:, i])
            # print('partner: %s  -valid_auc: %s' % (target[i], str(round(roc_auc_model[target[i]], 4))))

        return loss, acc, roc_auc_model

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        prediction = []
        true = []

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        self.model.eval()
        for i, (x, neibor_x, label) in enumerate(self.test_loader):
            if self.use_gpu:
                x, neibor_x, label = x.cuda(), neibor_x.cuda(), label.cuda()
            x, neibor_x, label = Variable(x), Variable(neibor_x), Variable(label)
            true.extend(label.data.numpy())

            # forward pass
            outputs = self.model(x, neibor_x)
            loss = self.loss_ce(outputs, label)

            prediction.extend(outputs.detach().numpy())

            # measure accuracy and record loss
            acc = accuracy(outputs.data, label.data)
            losses.update(loss.item(), x.size()[0])
            accs.update(acc, x.size()[0])

        print(
            '[*] Test loss: {:.3f}, acc: {:.3f}%'.format(
                losses.avg, accs.avg)
        )
        metrics(prediction, true)



    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth'
        if best:
            filename = self.model_name + '_model_best.pth'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )
