from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
import json
import torch as ch
import torchvision
import argparse
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField 

from ffcv.fields import IntField, RGBImageField


def main(train_dataset, val_dataset):
    datasets = {
        'train': torchvision.datasets.CIFAR10('/home/sourav', train=True, download=True),
        'test': torchvision.datasets.CIFAR10('/home/sourav', train=False, download=True)
        }

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
    '--config', default='config.json', type=str, help='config file')
    parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)
    config_train_dataset = config['train_dataset']
    config_test_dataset = config['test_dataset']

    main(config_train_dataset,config_test_dataset)