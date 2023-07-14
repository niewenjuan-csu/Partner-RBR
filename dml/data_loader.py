# -*- coding: utf-8 -*-

import numpy as np
import pickle
import joblib

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F


def loadfile(file):
    data = joblib.load(file)
    return data


def get_data(data_list):
    input = []
    neibor_input = []
    y = []

    row, col = data_list[0][0].shape[0], data_list[0][0].shape[1]
    row_n, col_n = data_list[0][1].shape[0], data_list[0][1].shape[1]
    for i in range(len(data_list)):
        input.append(data_list[i][0])
        neibor_input.append(data_list[i][1])
        y.append(data_list[i][2])

    # (winsize, seqfea, 1)
    input = np.array(input, dtype='float32').reshape((-1, row, col, 1))
    input = np.transpose(input, (0, 3, 1, 2))  # (batch, C, H, W)
    # (winsize, structure_fea, 1)
    neibor_input = np.array(neibor_input).reshape((-1, row_n, col_n, 1))
    neibor_input = np.transpose(neibor_input, (0, 3, 1, 2))
    y = np.array(y)
    print(input.shape)
    print(neibor_input.shape)
    print(y.shape)
    return input, neibor_input, y

class ResData(Dataset):
    def __init__(self, file):
        data = loadfile(file)
        x, neibor_x, y = get_data(data)
        self.x = torch.from_numpy(x)
        self.neibor_x = torch.from_numpy(neibor_x)
        self.y = torch.from_numpy(y).long()
        self.len = len(data)

    def __getitem__(self, item):
        # multi-class
        return self.x[item], self.neibor_x[item], self.y[item]

    def __len__(self):
        return self.len


def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=0,
                     pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.

    Returns
    -------
    - data_loader: train set iterator.
    """
    dataset = ResData(data_dir)
    if shuffle:
        np.random.seed(random_seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader



def get_test_loader(data_dir,
                    batch_size,
                    num_workers=0,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # load dataset
    dataset = ResData(data_dir)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return data_loader

if __name__ == '__main__':
    trainfile = '../data/win11/msa_lm_structure/fold1'
    # testfile = '../data/test_win11.pickle'

    traindata = ResData(trainfile)

    # testdata = TestData(testfile)
    traindata_loader =DataLoader(traindata, batch_size=256, shuffle=True)
    for i, data in enumerate(traindata_loader):
        x, neibor_x, y = data
        x, neibor_x, y = Variable(x), Variable(neibor_x), Variable(y)
        print(i, "个inputs", x.data.size(), "labels", y.data.size())


