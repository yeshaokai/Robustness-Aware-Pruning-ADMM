import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from alexnet_bn import AlexNet_BN
import numpy as np
from numpy import linalg as LA
from config import Config

sys.path.append('../../') # append root directory

from admm.warmup_scheduler import GradualWarmupScheduler
from admm.cross_entropy import CrossEntropyLossMaybeSmooth
from admm.utils import mixup_data, mixup_criterion
import admm 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--config_file', type=str, default='', help ="define config file")
parser.add_argument('--stage', type=str, default='', help ="select the pruning stage")
best_acc1 = 0


def main():
    args = parser.parse_args()
    config = Config(args)
    
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc1
    config.gpu = gpu

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
    # create model
    if config.pretrained:
        print("=> using pre-trained model '{}'".format(config.arch))

        model = models.__dict__[config.arch](pretrained=True)
        print(model)
        param_names = []
        module_names = []
        for name,W in model.named_modules():
            module_names.append(name)
        print (module_names)
        for name,W in model.named_parameters():
            param_names.append(name)
        print (param_names)
    else:
        print("=> creating model '{}'".format(config.arch))
        if config.arch == "alexnet_bn":
            model = AlexNet_BN()
            print (model)
            for i,(name,W) in enumerate(model.named_parameters()):
                print (name)
        else:
            model = models.__dict__[config.arch]()
            print (model)


    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int(config.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.arch.startswith('alexnet') or config.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
         
            model = torch.nn.DataParallel(model).cuda()
    config.model = model            
    # define loss function (criterion) and optimizer



    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config.smooth_eps).cuda(config.gpu)
    
    config.smooth = config.smooth_eps > 0.0
    config.mixup = config.alpha > 0.0

    # note that loading a pretrain model does not inherit optimizer info
    # will use resume to resume admm training
    if config.load_model:
        if os.path.isfile(config.load_model):
            if (config.gpu):
                model.load_state_dict(torch.load(config.load_model,map_location = {'cuda:0':'cuda:{}'.format(config.gpu)}))
            else:
                model.load_state_dict(torch.load(config.load_model))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))
            

    config.prepare_pruning()

    nonzero = 0
    zero = 0
    for name,W in model.named_parameters():
        if name in config.conv_names:
            W = W.cpu().detach().numpy()
            zero +=np.sum(W==0)
            nonzero+=np.sum(W!=0)
    total = nonzero+zero
    print ('compression rate is {}'.format(total*1.0/nonzero))
    import sys
    sys.exit()
        
    # optionally resume from a checkpoint
    if config.resume:
        ## will add logic for loading admm variables
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            
            ADMM.ADMM_U = checkpoint['admm']['ADMM_U']
            ADMM.ADMM_Z = checkpoint['admm']['ADMM_Z']
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    config.warmup = (not config.admm) and config.warmup_epochs > 0
    optimizer_init_lr = config.warmup_lr if config.warmup else config.lr

    optimizer = None
    if (config.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    elif (config.optimizer =='adam'):
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
        
    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs*len(train_loader), eta_min=4e-08)
    elif config.lr_scheduler == 'default':
        # sets the learning rate to the initial LR decayed by gamma every 30 epochs"""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30*len(train_loader), gamma=0.1)
    else:
        raise Exception("unknown lr scheduler")
    
    if config.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=config.lr/config.warmup_lr, total_iter=config.warmup_epochs*len(train_loader), after_scheduler=scheduler)

    if False:
        validate(val_loader,criterion,config)
        return
    ADMM = None
    
    if config.verify:
        admm.masking(config)
        admm.test_sparsity(config)
        validate(val_loader,criterion,config)
        import sys
        sys.exit()
    if config.admm:
        ADMM = admm.ADMM(config)

    if config.masked_retrain:
        # make sure small weights are pruned and confirm the acc
        admm.masking(config)
        print ("before retrain starts")    
        admm.test_sparsity(config)
        validate(val_loader, criterion, config)
    if config.masked_progressive:
        admm.zero_masking(config)  
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch

        train(train_loader,config,ADMM, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best and not config.admm: # we don't need admm to have best validation acc
            print ('saving new best model {}'.format(config.save_model))
            torch.save(model.state_dict(),config.save_model)                    
        
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                and config.rank % ngpus_per_node == 0):
            save_checkpoint(config,{'admm':{},
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
    # save last model for admm, optimizer detail is not necessary
    if config.save_model and config.admm:
        print ('saving model {}'.format(config.save_model))
        torch.save(model.state_dict(),config.save_model)        
    if config.masked_retrain:
        print ("after masked retrain")
        admm.test_sparsity(config)
def train(train_loader,config,ADMM, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    config.model.train()    
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if config.admm:
            admm.admm_adjust_learning_rate(optimizer,epoch,config)
        else:
            scheduler.step()

        input = input.cuda(config.gpu, non_blocking=True)
        target = target.cuda(config.gpu) 
        data = input

        if config.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, config.alpha)

        # compute output
        output = config.model(input)

        if config.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, config.smooth)
        else:
            ce_loss = criterion(output, target, smooth=config.smooth)

        if config.admm:
            admm.admm_update(config,ADMM,device,train_loader,optimizer,epoch,data,i)   # update Z and U
            ce_loss,admm_loss,mixed_loss = admm.append_admm_loss(config,ADMM,ce_loss) # append admm losss        

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

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
                        W.grad *=config.masks[name]                    
        
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
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)) 
            print ("cross_entropy loss: {}".format(ce_loss))           


def validate(val_loader, criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model = config.model
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):


            input = input.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(config,state, is_best, filename='checkpoint.pth.tar', ADMM = None):
    filename = 'checkpoint_gpu{}.pth.tar'.format(config.gpu)

    if config.admm and ADMM:  ## add admm variables to state
        state['admm']['ADMM_U'] = ADMM.ADMM_U
        state['admm']['ADMM_Z'] = ADMM.ADMM_Z
    
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


if __name__ == '__main__':
    main()
