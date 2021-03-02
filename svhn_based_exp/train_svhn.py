from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse

from data_prepare import svhn_transform_train, svhn_transform_test
from utils import get_num_classes, get_model, get_dataset, get_criterion, get_upstream_preprocessing, train_epoch, test_epoch, test_by_nn, split_test_val
from logger import set_logger

parser = argparse.ArgumentParser(description='Train CIFAR model')
parser.add_argument('--data_root', type=str, default='~/data', help='CIFAR data root')
parser.add_argument('--save_dir', type=str, default='checkpoint', help='model output directory')
parser.add_argument('--para_to_vary_model', type=float, default=None, help='r for loss-function mismatch experiment, p for data-distribution mismatch, p for upstream-upstream entanglement (anti-correlated error)')
parser.add_argument('--model', type=str, default='convnet', help='type of model')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--task', type=str, default="lf-mis-up", help='lf-mis-up / lf-mis-down / dd-mis-up / dd-mis-down / hs-mis-up / hs-mis-down / uu-ent-up1 / uu-ent-up1 / uu-ent-down')
parser.add_argument('--upstream_setting', type=str, default=None, help='up-i-setting={task}_{para_to_vary_model}_{model}_{seed}_{repre}_{model_specify} where repre is selected from {logits, preds, feat}, model_specify is selected from {last, best}. upstream_setting={up-1-setting}_..._{{up-M-setting}}')
args = parser.parse_args()

#create folders for saving logs and checkpoints
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.data_root, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "log"), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "checkpoint"), exist_ok=True)

#make the name for saving logs and checkpoints
save_name = f"svhn_{args.model}_task_{args.task}_upstream_setting_{args.upstream_setting}_para_to_vary_model_{args.para_to_vary_model}_seed_{args.seed}"
logger = set_logger("", f"{args.save_dir}/log/{save_name}.txt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print args
logger.info(args)

#transform the para from paper to code
if args.task == "lf-mis-up":
    args.para_to_vary_model = (args.para_to_vary_model / 10)  + 0.1
elif args.task in ["uu-ent-up1", "uu-ent-up2", "dd-mis-up"]:
    args.para_to_vary_model = args.para_to_vary_model / 100
    
#initialize datasets
trainset, testset = get_dataset(args.task, args.para_to_vary_model, args.seed, args.data_root)
torch.manual_seed(args.seed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

#process the dataset if this task has upstream(s)
if args.upstream_setting is not None:
    trainloader, testloader = get_upstream_preprocessing(trainloader, testloader, args.upstream_setting, args.save_dir, device)
    linear_base = trainloader.dataset.tensors[0].size()[-1]
else:
    linear_base = None #doesn't support linear model or MLP to original image data yet
    
#when args.task == lf_mis_up, split the testloader to testloader and valloader and use validation to select the model
if args.task == "lf-mis-up":
    testloader, valloader = split_test_val(testloader)

#get the model to be trained
num_classes = get_num_classes(args.task)
net = get_model(args.model, num_classes, device=device, linear_base=linear_base)

#initialize the optimizer and criterion
optimizer = optim.Adam(net.parameters(), lr=args.lr)
first_drop, second_drop = False, False
criterion = get_criterion(args.task) 

#initialize best checkpoints
best_state = None
best_loss = float("inf")

#make a bool variable to indicate whether it is a regression task
regression = args.task == "lf-mis-up"

for epoch in range(args.epochs):
    logger.info('\nEpoch: %d' % epoch)
    
    #train and test
    train_epoch(optimizer, criterion, trainloader, net, logger, regression, device)
    loss, loss_detail = test_epoch(criterion, testloader, net, logger, regression, device)
    knn_loss = None if regression else test_by_nn(testloader, net, args.model, logger, device)
    
    #update best state only for regression task (lf-mis-up)
    if regression:
        val_loss, val_loss_detail = test_epoch(criterion, valloader, net, logger, regression, device)
        if val_loss < best_loss:
            best_state = {
                        'net': net.state_dict(),
                        'epoch': epoch,
                        'loss': loss,
                        'loss_detail': loss_detail,
                        'val_loss': val_loss,
                        'val_loss_detail': val_loss_detail,
            }
            best_loss = val_loss
     
    #adjust learning rate
    if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        first_drop = True
    if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        second_drop = True

    #save the checkpoint
    if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'loss_detail': loss_detail,
            'knn_loss': knn_loss,
            'best_state': best_state
        }
        torch.save(state, f'{args.save_dir}/checkpoint/{save_name}.pth')
logger.info("save to: " + f'{args.save_dir}/checkpoint/{save_name}.pth')
