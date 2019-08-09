'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse
#from utils import *
from models import *
#from config import Config
#from resnet import ResNet18_adv
import numpy as np
from numpy import linalg as LA

sys.path.append('../../') # append root directory

from admm.warmup_scheduler import GradualWarmupScheduler
from admm.cross_entropy import CrossEntropyLossMaybeSmooth
from admm.utils import mixup_data, mixup_criterion
import admm

model_names = ['vgg16','resnet18','vgg16_1by8','vgg16_1by16','vgg16_1by32']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AttackPGD(nn.Module):
    def __init__(self, basic_model):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        #self.rand = config.random_start
        self.rand = True
        #self.step_size = config.step_size/255
        self.step_size = 2.0/255
        #self.epsilon = config.epsilon/255
        self.epsilon = 8.0/255
        #self.num_steps = config.num_steps
        self.num_steps = 10


    def forward(self,input, target):    # do forward in the module.py
        #if not args.attack :
        #    return self.basic_model(input), input



        x = input.detach()
        
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_model(x)
                loss = F.cross_entropy(logits, target, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, input - self.epsilon), input + self.epsilon)

            x = torch.clamp(x, 0, 1)

        return self.basic_model(input), self.basic_model(x) , x



def validate(model, val_loader,criterion, device):
    gpu = 0
    batch_time = AverageMeter()
    nat_losses = AverageMeter()
    adv_losses = AverageMeter()    
    nat_top1 = AverageMeter()
    adv_top1 = AverageMeter()    
    print_freq = 10
    nat_loss = 0
    adv_loss = 0
    pertb_output = []
    label_output = []

    # switch to evaluate mode
    model.eval()
    best_adv_acc = 0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            if gpu is not None:
                input = input.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            nat_output,adv_output,pert_inputs = model(input,target)
            pertb_output.append(pert_inputs.cpu().detach().numpy())
            label_output.append(target.cpu().detach().numpy())
            nat_loss = criterion(nat_output, target)
            adv_loss = criterion(adv_output, target)            

            # measure accuracy and record loss
            nat_acc1, nat_acc5 = accuracy(nat_output, target, topk=(1, 5))
            adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))
            nat_losses.update(nat_loss.item(), input.size(0))
            adv_losses.update(adv_loss.item(), input.size(0))            
            nat_top1.update(nat_acc1[0], input.size(0))
            adv_top1.update(adv_acc1[0], input.size(0))            


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Nat_Loss {nat_loss.val:.4f} ({nat_loss.avg:.4f})\t'
                      'Nat_Acc@1 {nat_top1.val:.3f} ({nat_top1.avg:.3f})\t'
                      'Adv_Loss {adv_loss.val:.4f} ({adv_loss.avg:.4f})\t'
                      'Adv_Acc@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})\t'                      
                      .format(
                       i, len(val_loader), batch_time=batch_time, nat_loss=nat_losses,
                          nat_top1=nat_top1,adv_loss=adv_losses,adv_top1=adv_top1))


        print(' * Nat_Acc@1 {nat_top1.avg:.3f} *Adv_Acc@1 {adv_top1.avg:.3f}'
              .format(nat_top1=nat_top1,adv_top1=adv_top1))

 #       global best_adv_acc
        if adv_top1.avg.item()>best_adv_acc :
            best_adv_acc = adv_top1.avg.item()
            print ('new best_adv_acc is {top1.avg:.3f}'.format(top1=adv_top1))
            #print ('saving model {}'.format(config.save_model))
            #torch.save(config.model.state_dict(),config.save_model)
    pertb_output = np.array(pertb_output)
    label_output = np.array(label_output)
    #np.save('pert_output_w16.npy', pertb_output)
    #np.save('label_output_w16.npy',label_output)
    return adv_top1.avg

    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar Example')

    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    transform_test = transforms.Compose([
    transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#    test_loader = torch.utils.data.DataLoader(
#        datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                           transforms.ToTensor()
#                           #transforms.Normalize((0.1307,), (0.3081,))
#                       ])),
#        batch_size=1000, shuffle=True, **kwargs)

    
    args = parser.parse_args()

    gpu = 0
    

        
    torch.manual_seed(1)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = "cpu"

    model = ResNet18_adv(w=1).to(device)
    model = AttackPGD(model)
    model = torch.nn.DataParallel(model)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=0.0).cuda(gpu) 
       
    load_model = 'resnet18_adv_w1.pt'
        
        # unlike resume, load model does not care optimizer status or start_epoch
    print('==> Loading from {}'.format(load_model))

    model.load_state_dict(torch.load(load_model,map_location = {'cuda:0':'cuda:{}'.format(gpu)}))
    


    
    print(model)


    validate(model, testloader,criterion, device)

    for name,W in model.named_parameters():

        W = W.cpu().detach().numpy()
        shape = W.shape
        W2d = W.reshape(shape[0],-1)
        column_l2_norm = LA.norm(W2d,2,axis=0)
        zero_column = np.sum(column_l2_norm == 0)
        nonzero_column = np.sum(column_l2_norm !=0)

        print ("column sparsity of layer {} is {}".format(name,zero_column/(zero_column+nonzero_column)))

    
    for name,W in model.named_parameters():
        W = W.cpu().detach().numpy()
        shape = W.shape
        W2d = W.reshape(shape[0],-1)
        row_l2_norm = LA.norm(W2d,2,axis=1)
        zero_row = np.sum(row_l2_norm == 0)
        nonzero_row = np.sum(row_l2_norm !=0)
        print ('filter sparsity of layer {} is {}'.format(name,zero_row/(zero_row+nonzero_row))) 

    

        
if __name__ == '__main__':
    main()
