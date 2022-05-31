""" Main module for testing optimizers """
# Load libraries and pick the CUDA device if available
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

# Custom libraries
from AdaptiveLinearCoupling import *
from AdaACSA import *
from AdaAGDplus import *
from AdaJRGS import *
from CustomOptimizer import *


from loader import *
from src.models import MODELS_MAP
from src.utils import *
#from misc import progress_bar

def flat_weight_dump(model):
    """ Returns a 1-d tensor containing all the network weights """
    is_empty = True
    for _, param in model.named_parameters():
        if param.requires_grad:
            if is_empty:
                flat_tensor = param.data.flatten()
                is_empty = False
            else:
                flat_tensor = torch.cat([flat_tensor, param.data.flatten()])
    return flat_tensor


def tb_dump(epoch, net, writer1,writer2):
    """ Routine for dumping info on tensor board at the end of an epoch """
    print('=> eval on test data')
    (test_loss, test_acc, _) = test(testloader, net, device)
    writer1.add_scalar('Loss/test', test_loss, epoch)
    writer1.add_scalar('Accuracy', test_acc, epoch)

    print('=> eval on train data')
    (train_loss, train_acc, _) = test(trainloader, net, device)
    writer2.add_scalar('Loss/train', train_loss, epoch)
    writer2.add_scalar('Accuracy', train_acc, epoch)
    print('epoch %d done\n' % (epoch))


def test(testloader, net, device):
    """ Routine for evaluating test error """
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            outputs = net(images.to(device))
            loss += criterion(outputs, labels.to(device)) * labels.size(0)
            # track total loss until now, not average loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            progress_bar(
                batch_idx, len(testloader), 'Loss: %.5f | Acc: %.3f%% (%d/%d)'
                % (loss/total, 100.*correct/total, correct, total))
    # Return the average loss (i.e. total loss averaged by number of samples)
    return (loss.item() / total, 100.0*correct/total, total)

# Trains the network
def train_net(epochs, path_name_test,path_name_train, net, optimizer):
    """ Train the network """
    print(optimizer)
    writer1 = SummaryWriter(path_name_test)
    writer2 = SummaryWriter(path_name_train)
    n_iter = 0
    
    # Dump info on the network before running any training step
    tb_dump(0, net, writer1,writer2)
    early_stopping = EarlyStopping()
    for epoch in range(epochs):  # Loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            #Plotting the weight distribution 
            #weight_histograms(writer2, epoch, net)

            # Get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Save weights before gradient step, in order to measure movement
            if config_dump_movement and (i % config_batch_statistics_freq == 0):
                old_weights = flat_weight_dump(net)

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()

            # Compute statistics
            train_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(device)).sum().item()
            train_acc = 100 * correct / labels.size(0)

            progress_bar(
                i, len(trainloader), 'Loss: %.5f | Acc: %.3f%%'
                % (train_loss, 100.*correct/labels.size(0)))

            # Print statistics every couple of mini-batches
            if i % config_batch_statistics_freq == 0:
                writer2.add_scalar('Loss/batch', train_loss, n_iter)
                writer2.add_scalar('Accuracy/batch', train_acc, n_iter)

                if config_dump_movement:
                    new_weights = flat_weight_dump(net)
                    movement = torch.norm(
                        torch.add(old_weights, new_weights, alpha=-1))
                    writer2.add_scalar('Movement', movement.item(), n_iter)

                writer2.flush()
                n_iter = n_iter + 1
	
#	early_stopping(test_loss)
        (test_loss, test_acc, _) = test(testloader, net, device)
        #Save the checkpoint
        '''
        if(epoch%(20)==0):
          checkpoint = {
            'epoch': epoch ,
            'state_dict': net.state_dict(),
            #'optimizer': optimizer.state_dict(),
          }
        
           
          save_ckp(checkpoint, f'checkpoints1/epoch_model_{epoch}.pth')

        '''
        tb_dump(epoch+1, net, writer1,writer2)
        
        if config_e_stop == 1:
            if epoch > config_epoch_for_estop:
                early_stopping(test_loss)
                if early_stopping.early_stop:
                    break
    print('Finished Training')
    writer2.close()
    writer1.close()


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
config_e_stop = config["e_stop"]
config_epoch_for_estop = config["epoch_for_estop"]
config_one_shot_prune = config["one_shot_prune"]
config_iterative_prune = config["iterative_prune"]
config_epochs_to_finetune = config["epochs_to_finetune"]
config_epochs_to_densetrain = config["epochs_to_densetrain"]
#config_initial_accumulator_value = config['initial_accumulator_value']
#config_beta = config['beta']
#config_eps = config['eps']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Set random seed
torch.manual_seed(config_random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config_random_seed)

# Load data
if config_dataset == 'MNIST':
    trainloader, testloader = mnist_loader(batch_size=config_batch_size)
else:
    trainloader, testloader = cifar_loader(batch_size=config_batch_size)

model = MODELS_MAP[config_architecture]()
net = model.to(device)
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
    len_step = len(trainloader),
    
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

train_net(
    epochs=config_epochs, path_name_test=config_tb_path_test,path_name_train=config_tb_path_train, net=net, optimizer=optimizer)

# Dump some info on the range of parameters after training is finished
for param in net.parameters():
    print(str(torch.min(param.data).item()) + " " + str(torch.max(param.data).item()))
