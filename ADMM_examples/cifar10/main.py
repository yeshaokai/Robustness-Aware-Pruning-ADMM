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
from utils import *
from models import *
from config import Config



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



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config_file', type=str, default='', help ="config file")
parser.add_argument('--stage', type=str, default='', help ="select the pruning stage")


args = parser.parse_args()

config = Config(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if config.logging:
    log_dir = config.log_dir
    logger = getLogger(log_dir)
    logger.info(json.dumps(config.__dict__, indent=4))
else:
    logger = None


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=config.workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=config.workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Model
print('==> Building model..')
model = None
if config.arch == "vgg16":
    model = VGG('vgg16', w= config.width_multiplier)
elif config.arch =="resnet18":
    model = ResNet18()
elif config.arch == "googlenet":
    model = GoogLeNet()
elif config.arch == "densenet121":
    model = DenseNet121()
elif config.arch == "vgg16_1by8":
    model = VGG('vgg16_1by8')
elif config.arch == "vgg16_1by16":
    model = VGG('vgg16_1by16')
elif config.arch == "vgg16_1by32":
    model = VGG('vgg16_1by32')
elif config.arch == "resnet18_1by16":
    model = ResNet18_1by16()
elif config.arch == 'lenet':
    model = LeNet(w = config.width_multiplier)
# model = PreActResNet18()
# model = GoogLeNet()
# model = DenseNet121()
# model = ResNeXt29_2x64d()
# model = MobileNet()
# model = MobileNetV2()
# model = DPN92()
# model = ShuffleNetG2()
# model = SENet18()
# model = ShuffleNetV2(1)
print (model)

config.model = model

if device == 'cuda':
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        config.model = torch.nn.DataParallel(model,device_ids = [config.gpu])
    else:
        config.model.cuda()
        config.model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if config.load_model:
    # unlike resume, load model does not care optimizer status or start_epoch
    print('==> Loading from {}'.format(config.load_model))

    config.model.load_state_dict(torch.load(config.load_model)) # i call 'net' "model"
    


    
config.prepare_pruning() # take the model and prepare the pruning

ADMM = None

if config.admm:
    ADMM = admm.ADMM(config)



if config.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    config.model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    ADMM.ADMM_U = checkpoint['admm']['ADMM_U']
    ADMM.ADMM_Z = checkpoint['admm']['ADMM_Z']

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
config.smooth = config.smooth_eps > 0.0
config.mixup = config.alpha > 0.0


config.warmup = (not config.admm) and config.warmup_epochs > 0
optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

optimizer = None
if (config.optimizer == 'sgd'):
    optimizer = torch.optim.SGD(config.model.parameters(), optimizer_init_lr,
                            momentum=0.9,
                                weight_decay=1e-4)
elif (config.optimizer =='adam'):
    optimizer = torch.optim.Adam(config.model.parameters(), optimizer_init_lr)



scheduler = None
if config.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs*len(trainloader), eta_min=4e-08)
elif config.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    epoch_milestones = [150, 250, 350]

    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(trainloader) for i in epoch_milestones], gamma=0.1)
else:
    raise Exception("unknown lr scheduler")

if config.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier=config.lr/config.warmup_lr, total_iter=config.warmup_epochs*len(trainloader), after_scheduler=scheduler)


def train(train_loader,criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    config.model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, config)
        else:
            scheduler.step()

        if config.gpu is not None:
            input = input.cuda(config.gpu, non_blocking=True)
        target = target.cuda(config.gpu, non_blocking=True)

        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)

        # compute output
        output = config.model(input)

        if config.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, config.smooth)
        else:
            ce_loss = criterion(output, target, smooth=config.smooth)

        if config.admm:
            admm.admm_update(config,ADMM,device,train_loader,optimizer,epoch,input,i)   # update Z and U
            ce_loss,admm_loss,mixed_loss = admm.append_admm_loss(config,ADMM,ce_loss) # append admm losss

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
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
                            W.grad *= config.masks[name] #returns boolean array called mask when weights are above treshhold

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))



def validate(val_loader,criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    config.model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = config.model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))
        global best_acc
        if top1.avg.item()>best_acc and not config.admm:
            best_acc = top1.avg.item()
            print ('new best_acc is {top1.avg:.3f}'.format(top1=top1))
            print ('saving model {}'.format(config.save_model))
            torch.save(config.model.state_dict(),config.save_model)

    return top1.avg


if config.admm:
    validate(testloader,criterion,config)

if config.masked_retrain:
    # make sure small weights are pruned and confirm the acc
    print ("<============masking both weights and gradients for retrain")
    admm.masking(config)
    print ("<============testing sparsity before retrain")
    admm.test_sparsity(config)
    validate(testloader,criterion,config)
if config.masked_progressive:
    admm.zero_masking(config)

for epoch in range(start_epoch, start_epoch+config.epochs):
    train(trainloader,criterion,optimizer,epoch,config)
    validate(testloader,criterion,config)

####LOG HERE###
if config.logging:
    logger.info(f'---Final Results---')
    logger.info(f'overall best_acc is {best_acc}')

print ('overall  best_acc is {}'.format(best_acc))


if config.masked_retrain:
    print ("<=====confirm sparsity")
    admm.test_sparsity(config)


if config.save_model and config.admm:
    print ('saving model {}'.format(config.save_model))
    torch.save(config.model.state_dict(),config.save_model)
