# -*- coding: utf-8 -*-

import torch

from dml.Trainer import Trainer
from dml.config import get_config
from dml.utils import prepare_dirs, save_config
from dml.data_loader import get_train_loader, get_test_loader
from dml.Ind_Trainer import Ind_Trainer


def DML_main(config):
    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        # torch.backends.cudnn.deterministic = True

    # instantiate data loaders
    if config.is_train:
        train_data_loader = get_train_loader(
            config.data_dir+'/train_balance.pkl', config.batch_size,
            config.random_seed, config.shuffle, **kwargs
        )
        test_data_loader = get_test_loader(
            config.data_dir+'/valid.pkl', config.batch_size, **kwargs
        )
        data_loader = (train_data_loader, test_data_loader)
    else:
        test_data_loader = get_test_loader(
            config.data_dir+'/test.pkl', config.batch_size, **kwargs
        )
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()

def Ind_main(config):
    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        # torch.backends.cudnn.deterministic = True

    if config.is_train:
        train_data_loader = get_train_loader(
            config.data_dir + '/train_balance.pkl', config.batch_size,
            config.random_seed, config.shuffle, **kwargs
        )
        test_data_loader = get_test_loader(
            config.data_dir + '/valid.pkl', config.batch_size, **kwargs
        )
        data_loader = (train_data_loader, test_data_loader)
    else:
        test_data_loader = get_test_loader(
            config.data_dir + '/test.pkl', config.batch_size, **kwargs
        )
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Ind_Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()



if __name__ == '__main__':
    config, unparsed = get_config()
    DML_main(config)
    # Ind_main(config)