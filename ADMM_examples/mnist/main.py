from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import linalg as LA
import sys
from lenet_bn import LeNet_BN
from lenet import LeNet
from lenet_adv import LeNet_adv
from config import Config



sys.path.append("../../") # append root directory

from admm.warmup_scheduler import GradualWarmupScheduler
from admm.cross_entropy import CrossEntropyLossMaybeSmooth
from admm.utils import mixup_data, mixup_criterion
import admm 

best_acc = 0

model_names = ['lenet','lenet_bn','lenet_adv']


def save_checkpoint(config,state,filename='checkpoint.pth.tar'):
    torch.save(state,filename)
    
    

def train(config,ADMM,device,train_loader, criterion, optimizer, scheduler, epoch):
    config.model.train()

    ce_loss = None
    for batch_idx, (data, target) in enumerate(train_loader):

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            if scheduler is not None:
                scheduler.step()

        data, target = data.to(device), target.to(device)
        if config.gpu is not None:
            data = data.cuda(config.gpu, non_blocking = True)
            target = target.cuda(config.gpu,non_blocking = True)

        if config.mixup:
            data, target_a, target_b, lam = mixup_data(data, target, config.alpha)

        optimizer.zero_grad()
        output = config.model(data)

        if config.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, config.smooth)
        else:
            ce_loss = criterion(output, target, smooth=config.smooth)
        
        if config.admm:
            admm.admm_update(config,ADMM,device,train_loader,optimizer,epoch,data,batch_idx)   # update Z and U        
            ce_loss,admm_loss,mixed_loss = admm.append_admm_loss(config,ADMM,ce_loss) # append admm losss

        
        if config.admm:
            mixed_loss.backward()
        else:
            ce_loss.backward()

        if config.masked_progressive:
            with torch.no_grad():            
                for name,W in config.model.named_parameters():
                    if name in config.zero_masks:
                            W.grad *=config.zero_masks[name]


        if config.masked_retrain:
            with torch.no_grad():
                for name,W in config.model.named_parameters():
                    if name in config.masks:
                            W.grad *=config.masks[name]
            
        optimizer.step()
        if batch_idx % config.print_freq == 0:

             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), ce_loss.item()))
    


def test(config, device, test_loader):
    config.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if config.gpu is not None:
                data = data.cuda(config.gpu, non_blocking = True)
                target = target.cuda(config.gpu,non_blocking = True)
            output = config.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct /len(test_loader.dataset)
    global best_acc
    if acc > best_acc and not config.admm:
        best_acc = acc
        print ('new best acc is {}'.format(best_acc))
        print ('saving model {}'.format(config.save_model))
        torch.save(config.model.state_dict(),config.save_model)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config_file', type=str, default='', help ="config file")
    parser.add_argument('--stage', type=str, default='', help ="select the pruning stage")

    
    args = parser.parse_args()

    config = Config(args)
    
    use_cuda = True
        
    torch.manual_seed(1)
    
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)


    

    model = None
    if config.arch == 'lenet_bn':
        model = LeNet_BN().to(device)
    elif config.arch == 'lenet':
        model = LeNet().to(device)
    elif config.arch == 'lenet_adv':
        model = LeNet_adv(config.width_multiplier).to(device)
    torch.cuda.set_device(config.gpu)
    model.cuda(config.gpu)
    
    config.model = model


    
    ADMM = None

    config.prepare_pruning()
    
    if config.admm:
        ADMM = admm.ADMM(config)

    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
    config.smooth = config.smooth_eps > 0.0
    config.mixup = config.alpha > 0.0
    
    
    config.warmup = (not config.admm) and config.warmup_epochs > 0
    optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

    if (config.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(config.model.parameters(), optimizer_init_lr,
                                momentum=0.9,
                                    weight_decay=1e-4)
    elif (config.optimizer =='adam'):
        optimizer = torch.optim.Adam(config.model.parameters(), optimizer_init_lr)    
    else:
        raise Exception("unknown optimizer")

    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs*len(train_loader), eta_min=4e-08)
    elif config.lr_scheduler == 'default':
        pass
    else:
        raise Exception("unknown lr scheduler")

        
    if config.load_model:
        # unlike resume, load model does not care optimizer status or start_epoch
        print('==> Loading from {}'.format(config.load_model))
        config.model.load_state_dict(torch.load(config.load_model,map_location = {'cuda:0':'cuda:{}'.format(config.gpu)}))
        test(config,  device, test_loader) 

    global best_acc
    if config.resume:
        if os.path.isfile(config.resume):
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))            


    
            
    if config.masked_retrain:
        # make sure small weights are pruned and confirm the acc
        print ("<============masking both weights and gradients for retrain")    
        admm.masking(config)
        print ("<============testing sparsity before retrain")
        admm.test_sparsity(config)        
        test(config,  device, test_loader)        
    if config.masked_progressive:
        admm.zero_masking(config)
    
        
    for epoch in range(0, config.epochs+1):

        train(config,ADMM,device, train_loader, criterion, optimizer, scheduler, epoch)
        test(config, device, test_loader)
        save_checkpoint(config,{
            'epoch':epoch+1,
            'arch':config.arch,
            'state_dict':config.model.state_dict(),
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()})

    print ('overall  best_acc is {}'.format(best_acc))

    if (config.save_model and config.admm):
        print ('saving model {}'.format(config.save_model))
        torch.save(config.model.state_dict(),config.save_model)

        
if __name__ == '__main__':
    main()
