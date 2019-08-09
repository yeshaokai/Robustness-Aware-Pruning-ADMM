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


sys.path.append("../../") # append root directory



def test(model, device, test_loader):
    gpu = 0
    model.eval()
    nat_test_loss = 0
    nat_correct = 0
    adv_test_loss = 0
    adv_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if gpu is not None:
                data = data.cuda(gpu, non_blocking = True)
                target = target.cuda(gpu,non_blocking = True)                                    
            nat_output, adv_output ,pert_inputs = model(data,target)

            nat_test_loss += F.nll_loss(nat_output, target, reduction='sum').item() # sum up batch loss
            nat_pred = nat_output.max(1, keepdim=True)[1] # get the index of the max log-probability
            nat_correct += nat_pred.eq(target.view_as(nat_pred)).sum().item()
            
            
            adv_test_loss += F.nll_loss(adv_output, target, reduction='sum').item() # sum up batch loss
            adv_pred = adv_output.max(1, keepdim=True)[1] # get the index of the max log-probability
            adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum().item()

    nat_test_loss /= len(test_loader.dataset)            
    adv_test_loss /= len(test_loader.dataset)

    adv_acc = 100. * adv_correct /len(test_loader.dataset)
    nat_acc = 100. * nat_correct /len(test_loader.dataset)
    print('\nTest set: Average nat_loss: {:.4f}, nat_Accuracy: {}/{} ({:.2f}%)\n'.format(
        nat_test_loss, nat_correct, len(test_loader.dataset),
        100. * nat_correct / len(test_loader.dataset)))

    print('\nTest set: Average adv_loss: {:.4f}, adv Accuracy: {}/{} ({:.2f}%)\n'.format(
        adv_test_loss, adv_correct, len(test_loader.dataset),
        100. * adv_correct / len(test_loader.dataset)))


class AttackPGD(nn.Module):
    def __init__(self, basic_model):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = True
        self.step_size = 0.01
        self.epsilon = 0.3
        self.num_steps = 40


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

    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    use_cuda = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)

    
    args = parser.parse_args()

    gpu = 0
    

        
    torch.manual_seed(1)
    
    device = torch.device("cuda" if use_cuda else "cpu")


    model = LeNet_adv(w=16).to(device)
    model = AttackPGD(model)
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
        
    load_model = 'lenet_adv_retrained_w16_pruned.pt'
        
        # unlike resume, load model does not care optimizer status or start_epoch
    print('==> Loading from {}'.format(load_model))

    model.load_state_dict(torch.load(load_model,map_location = {'cuda:0':'cuda:{}'.format(gpu)}))
    
    print (model)

    test(model,device,test_loader)

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
