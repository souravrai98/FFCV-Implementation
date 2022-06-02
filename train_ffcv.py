import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision


from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from src.models import MODELS_MAP
from loader import *
from src.utils import *

from CustomOptimizer import *

def tb_dump(epoch, net,loaders, writer1,writer2):
    """ Routine for dumping info on tensor board at the end of an epoch """
    print('=> eval on test data')
    (test_loss, test_acc, _) = evaluate(net,loaders,'test')
    writer1.add_scalar('Loss/test', test_loss, epoch)
    writer1.add_scalar('Accuracy', test_acc, epoch)

    print('=> eval on train data')
    (train_loss, train_acc, _) = evaluate(net,loaders,'train')
    writer2.add_scalar('Loss/train', train_loss, epoch)
    writer2.add_scalar('Accuracy', train_acc, epoch)
    print('epoch %d done\n' % (epoch))

def train(model, loaders,epochs,optimizer,lr_peak_epoch):
    opt = optimizer
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    
    writer1 = SummaryWriter(config_tb_path_train)
    writer2 = SummaryWriter(config_tb_path_test)
      
    for epoch in range(epochs):

        model.train()
        with tqdm(loaders['train'], unit="batch") as tepoch:  
        
            
            for ims, labs in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                opt.zero_grad(set_to_none=True)
        
                with autocast():
        
                    out = model(ims)
        
                    loss = criterion(out, labs)

        
                scaler.scale(loss).backward()
        
                scaler.step(opt)
        
                scaler.update()
        
                scheduler.step()

                tepoch.set_postfix(loss=loss.item())
        tb_dump(epoch,model,loaders,writer1,writer2)

    print('Finished Training')
    writer2.close()
    writer1.close()


def evaluate(model, loaders,name):
    model.eval()
    
    with ch.no_grad():
        
        if name == 'train'  :  
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} Accuracy, {total_correct / total_num * 100:.1f}%, {name} Loss, {100 - (total_correct / total_num * 100):.2f}')
        else: 
            total_correct, total_num = 0., 0.      
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} Accuracy, {total_correct / total_num * 100:.1f}%, {name} Loss, {100 - (total_correct / total_num * 100):.2f}')
 
    return (100 - (total_correct / total_num * 100), total_correct / total_num * 100, total_num)
    


# ################
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
    '--config', default='config.json', type=str, help='config file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)
config_experiment_number = config['experiment_number']
config_dataset = config['dataset']
config_architecture = config['architecture']
config_batch_size = config['batch_size']
config_optimizer = config['optimizer']
config_lr = config['lr']
config_momentum = config['momentum']
config_prune_epoch = config["prune_epoch"]
config_unfreeze_epoch = config["unfreeze_epoch"]
config_perc_to_prune = config['perc_to_prune']
config_step_of_prune = config["step_of_prune"]
config_radius = config['radius']
config_epochs = config['epochs']
config_tb_path_test = config['tb_path_test']
config_tb_path_train = config['tb_path_train']
config_batch_statistics_freq = config['batch_statistics_freq']
config_dump_movement = bool(config['dump_movement'] == 1)
config_projected = bool(config['projected'] == 1)
config_weight_decay = config['weight_decay']
config_radius = config['radius']
config_random_seed = config['random_seed']
config_gamma0 = config['gamma0']

config_one_shot_prune = config["one_shot_prune"]
config_iterative_prune = config["iterative_prune"]
config_epochs_to_finetune = config["epochs_to_finetune"]
config_epochs_to_densetrain = config["epochs_to_densetrain"]
#config_initial_accumulator_value = config['initial_accumulator_value']
config_beta = config['beta']
#config_eps = config['eps']
config_train_dataset = config["train_dataset"]
config_test_dataset = config["test_dataset"]

loaders, start_time = make_dataloaders(config_train_dataset,config_test_dataset,
batch_size = config_batch_size,num_workers= 8)


model = MODELS_MAP[config_architecture]()
net = model.to(memory_format=ch.channels_last).cuda()
criterion = nn.CrossEntropyLoss()

if config_optimizer == 0:
    optimizer = optim.SGD(
      net.parameters(), lr=config_lr,
      momentum=config_momentum, weight_decay=config_weight_decay)
elif config_optimizer == 1:
    optimizer = optim.Adagrad(
      net.parameters(), lr=config_lr, weight_decay=config_weight_decay)
elif config_optimizer == 2:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=0, weight_decay=config_weight_decay)
elif config_optimizer == 3:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=1, weight_decay=config_weight_decay)
elif config_optimizer == 4:
    optimizer = optim.RMSprop(net.parameters(), lr=config_lr)
elif config_optimizer == 5:
    optimizer = AdaptiveLinearCoupling(
        net.parameters(), lr=config_lr,
        weight_decay=config_weight_decay)
elif config_optimizer == 6:
    #optimizer = AdaACSA(
    #    net.parameters(), lr=config_lr, radius=1, projected=config_projected)
    optimizer = AdaACSA(
        net.parameters(), lr=config_lr, radius=config_radius,
        weight_decay=config_weight_decay, projected=config_projected,
        gamma0=config_gamma0, beta=config_beta,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 7:
    optimizer = AdaAGDplus(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 8:
    optimizer = AdaJRGS(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)

elif config_optimizer == 9:
    optimizer = CustomOptimizer(net.parameters(),lr=config_lr, 
    momentum=config_momentum,
    weight_decay=config_weight_decay,
    len_step = len(loaders['train']),
    
    one_shot_prune  = config_one_shot_prune,
    prune_epoch=config_prune_epoch,
    step_of_prune=config_step_of_prune,
    perc_to_prune = config_perc_to_prune,

    iterative_prune = config_iterative_prune,
    unfreeze_epoch=config_unfreeze_epoch,
    epochs_to_densetrain = config_epochs_to_densetrain,
    epochs_to_finetune= config_epochs_to_finetune
   )

# Writer path for display on TensorBoard
if not os.path.exists(config_tb_path_test):
    os.makedirs(config_tb_path_test)
if not os.path.exists(config_tb_path_train):
    os.makedirs(config_tb_path_train)
#path_name = config_tb_path + \
#    str(config_experiment_number) + "_" + str(optimizer)

# Initialize weights
net.apply(weights_init_uniform_rule)

#load the model
#ckp_path = 'checkpoints/epoch_model_40.pth'
#checkpoint_model, start_epoch = load_ckp(ckp_path, net)





train(
    net, loaders, epochs=config_epochs,
          optimizer = optimizer,lr_peak_epoch=config_beta)
print(f'Total time: {time.time() - start_time:.5f}')
#evaluate(net, loaders)
