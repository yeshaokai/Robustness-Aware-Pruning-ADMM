from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from numpy import linalg as LA
import datetime
from tensorboardX import SummaryWriter
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class ADMM:
     def __init__(self,config):
          self.ADMM_U = {}
          self.ADMM_Z = {}
          self.model = config.model
          self.prune_ratios = None    #code name -> prune ratio
          self.init(config)
          
     def init(self,config):
          """
          Args:
              config: configuration file that has settings for prune ratios, rhos
          called by ADMM constructor. config should be a .yaml file          

          """          
          self.prune_ratios = config.prune_ratios
          self.rhos = config.rhos
          
          self.sparsity_type = config.sparsity_type
          for (name,W) in config.model.named_parameters():
              if name not in config.prune_ratios:
                  continue
              self.ADMM_U[name] = torch.zeros(W.shape).cuda() # add U 
              self.ADMM_Z[name] = torch.Tensor(W.shape).cuda() # add Z
                        
               


def weight_pruning(config,weight,prune_ratio):
     """ 
     weight pruning [irregular,column,filter]
     Args: 
          weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
          prune_ratio (float between 0-1): target sparsity of weights
     
     Returns:
          mask for nonzero weights used for retraining
          a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero 

     """     

     weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy     
    
     percent = prune_ratio * 100          
     if (config.sparsity_type == "irregular"):
         weight_temp = np.abs(weight)   # a buffer that holds weights with absolute values     
         percentile = np.percentile(weight_temp,percent)   # get a value for this percentitle
         under_threshold = weight_temp<percentile     
         above_threshold = weight_temp>percentile     
         above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
         weight[under_threshold] = 0     
         return torch.from_numpy(above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (config.sparsity_type == "column"):
          shape = weight.shape          
          weight2d = weight.reshape(shape[0],-1)
          shape2d = weight2d.shape
          column_l2_norm = LA.norm(weight2d,2,axis = 0)
          percentile = np.percentile(column_l2_norm,percent)
          under_threshold = column_l2_norm<percentile
          above_threshold = column_l2_norm>percentile
          weight2d[:,under_threshold] = 0
          above_threshold = above_threshold.astype(np.float32)
          expand_above_threshold = np.zeros(shape2d,dtype=np.float32)          
          for i in range(shape2d[1]):
               expand_above_threshold[:,i] = above_threshold[i]
          expand_above_threshold = expand_above_threshold.reshape(shape)
          weight = weight2d.reshape(shape)          
          return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (config.sparsity_type =="filter"):
          shape = weight.shape
          weight2d = weight.reshape(shape[0],-1)
          shape2d = weight2d.shape
          row_l2_norm = LA.norm(weight2d,2,axis = 1)
          percentile = np.percentile(row_l2_norm,percent)
          under_threshold = row_l2_norm <percentile
          above_threshold = row_l2_norm >percentile
          weight2d[under_threshold,:] = 0          
          above_threshold = above_threshold.astype(np.float32)
          expand_above_threshold = np.zeros(shape2d,dtype=np.float32)          
          for i in range(shape2d[0]):
               expand_above_threshold[i,:] = above_threshold[i]

          weight = weight2d.reshape(shape)
          expand_above_threshold = expand_above_threshold.reshape(shape)
          return torch.from_numpy(expand_above_threshold).cuda(),torch.from_numpy(weight).cuda()
     elif (config.sparsity_type =="bn_filter"):
          ## bn pruning is very similar to bias pruning
          weight_temp = np.abs(weight)
          percentile = np.percentile(weight_temp,percent)
          under_threshold = weight_temp<percentile     
          above_threshold = weight_temp>percentile     
          above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
          weight[under_threshold] = 0     
          return torch.from_numpy(above_threshold).cuda(),torch.from_numpy(weight).cuda()
     else:
          raise SyntaxError("Unknown sparsity type")
                                         
def test_sparsity(config):
     """
     test sparsity for every involved layer and the overall compression rate

     """
     total_zeros = 0
     total_nonzeros = 0

     print ('<===sparsity type is {}'.format(config.sparsity_type))
     print ('<===layers to be pruned are {}'.format(config._prune_ratios))
     if config.masked_progressive and (config.sparsity_type == 'filter' or config.sparsity_type =='column'or config.sparsity_type == "bn_filter" ):
         ### test both column and row sparsity
        print ("***********checking column sparsity*************")
        for name,W in config.model.named_parameters():
            if  name not in config.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0],-1)
            column_l2_norm = LA.norm(W2d,2,axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm !=0)

            print ("column sparsity of layer {} is {}".format(name,zero_column/(zero_column+nonzero_column)))
        print ("***********checking filter sparsity*************")            
        for name,W in config.model.named_parameters():
             if name not in config.prune_ratios:
                 continue
             W = W.cpu().detach().numpy()
             shape = W.shape
             W2d = W.reshape(shape[0],-1)
             row_l2_norm = LA.norm(W2d,2,axis=1)
             zero_row = np.sum(row_l2_norm == 0)
             nonzero_row = np.sum(row_l2_norm !=0)
             print ("filter sparsity of layer {} is {}".format(name,zero_row/(zero_row+nonzero_row)))
        print ("************checking overall sparsity in conv layers*************")
        for name,W in config.model.named_parameters():
            if  name not in config.prune_ratios:
                continue
            W = W.cpu().detach().numpy()            
            total_zeros +=np.sum(W==0)
            total_nonzeros +=np.sum(W!=0)
        print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))
        return
    
     if config.sparsity_type == "irregular":
         for name,W in config.model.named_parameters():
              if 'bias' in name:
                   continue
              W = W.cpu().detach().numpy()
              zeros = np.sum(W==0)
              total_zeros+=zeros
              nonzeros = np.sum(W!=0)
              total_nonzeros+=nonzeros
              print ("sparsity at layer {} is {}".format(name,zeros/(zeros+nonzeros)))
         total_weight_number = total_zeros+total_nonzeros
         print ('overal compression rate is {}'.format(total_weight_number/total_nonzeros))
     elif config.sparsity_type == "column":
        for name,W in config.model.named_parameters():
            if  name not in config.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0],-1)
            column_l2_norm = LA.norm(W2d,2,axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm !=0)
            total_zeros +=np.sum(W==0)
            total_nonzeros +=np.sum(W!=0)
            print ("column sparsity of layer {} is {}".format(name,zero_column/(zero_column+nonzero_column)))
        print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))          
     elif config.sparsity_type == "filter":
         print ('inside if')
         print (config.prune_ratios)
         for name,W in config.model.named_parameters():
             if name not in config.prune_ratios:
                 continue
             W = W.cpu().detach().numpy()
             shape = W.shape
             W2d = W.reshape(shape[0],-1)
             row_l2_norm = LA.norm(W2d,2,axis=1)
             zero_row = np.sum(row_l2_norm == 0)
             nonzero_row = np.sum(row_l2_norm !=0)
             total_zeros +=np.sum(W==0)
             total_nonzeros +=np.sum(W!=0)
             print ("filter sparsity of layer {} is {}".format(name,zero_row/(zero_row+nonzero_row)))
         print ('only consider conv layers, compression rate is {}'.format((total_zeros+total_nonzeros)/total_nonzeros))
     elif config.sparsity_type == "bn_filter":
          print ('inside bn_filter')
          print (config.prune_ratios)
          for i,(name,W) in enumerate(config.model.named_parameters()):
               if name not in config.prune_ratios:
                    continue
               W = W.cpu().detach().numpy()
               zeros = np.sum(W==0)
               nonzeros = np.sum(W!=0)
               print ("sparsity at layer {} is {}".format(name,zeros/(zeros+nonzeros)))



def predict_sparsity(config):
    # given a model, calculate the sparsity before proceeding.
    model = config.model
    total_parameters = 0 # parameters from  all conv layers
    nonzero_parameters = 0 # all remained non zero parameters
    layers = []
    ratios = []
    for name,W in model.named_parameters():
        if name not in config.prune_ratios:
            continue
        layers.append(W.cpu().detach().numpy())
        ratios.append(config.prune_ratios[name])
    for i in range(len(layers)):
        W = layers[i]
        ratio = ratios[i]
        numel = W.flatten().size
        total_parameters+=numel
        cur_nonzero = (1-ratio)*numel
        if i!=0 and ratios[i-1]!=0:
            cur_nonzero*=(1-ratios[i-1])
        nonzero_parameters += cur_nonzero            
    print ('predicting sparsity after pruning..... {}'.format(total_parameters/nonzero_parameters))
def admm_initialization(config,ADMM):
     if not config.admm:
          return
     for name,W in config.model.named_parameters():
          if name in ADMM.prune_ratios:
               _,updated_Z = weight_pruning(config,W,ADMM.prune_ratios[name]) # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
               ADMM.ADMM_Z[name] = updated_Z


def admm_update(config,ADMM,device,train_loader,optimizer,epoch,data,batch_idx):
     if not config.admm:
         return
     # sometimes the start epoch is not zero. It won't be valid if the start epoch is not 0
     if epoch == 0 and batch_idx == 0:
         admm_initialization(config,ADMM)  # intialize Z, U variable
     if epoch != 0 and epoch % config.admm_epoch == 0 and batch_idx == 0:
         for name,W in config.model.named_parameters():

             if name not in ADMM.prune_ratios:
                 continue

             if config.multi_rho:
                 admm_multi_rho_scheduler(ADMM,name) # call multi rho scheduler every admm update
             ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name] # Z(k+1) = W(k+1)+U[k]

             _, _Z = weight_pruning(config,ADMM.ADMM_Z[name],ADMM.prune_ratios[name]) #  equivalent to Euclidean Projection
             ADMM.ADMM_Z[name] = _Z

             ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name]+ ADMM.ADMM_U[name] # U(k+1) = W(k+1) - Z(k+1) +U(k)
                 



def append_admm_loss(config,ADMM,ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    ''' 
    admm_loss = {}
    
    if config.admm:
        if config.sparsity_type !="quantization":
            for name,W in config.model.named_parameters():  ## initialize Z (for both weights and bias)
                if name not in ADMM.prune_ratios:
                    continue
                
                admm_loss[name] = 0.5*ADMM.rhos[name]*(torch.norm(W-ADMM.ADMM_Z[name]+ADMM.ADMM_U[name],p=2)**2)
        else:
            for name,W in config.model.named_parameters():
                if name not in ADMM.number_bits:
                    continue
                admm_loss[name] = 0.5*ADMM.rhos[name]*(torch.norm(W-ADMM.alpha[name]*ADMM.ADMM_Q[name]+ADMM.ADMM_U[name],p=2)**2)
        mixed_loss = 0
        mixed_loss += ce_loss
        for k,v in admm_loss.items():
             mixed_loss+=v
        return ce_loss,admm_loss,mixed_loss

def admm_multi_rho_scheduler(ADMM,name):
    """
    It works better to make rho monotonically increasing
    
    """
    ADMM.rhos[name]*=1.3  # choose whatever you like

def admm_adjust_learning_rate(optimizer,epoch,config):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = config.admm_epoch
    lr = None
    if epoch % admm_epoch == 0:
         lr = config.lr
    else:
         admm_epoch_offset = epoch%admm_epoch

         admm_step = admm_epoch/3  # roughly every 1/3 admm_epoch. 
         
         lr = config.lr *(0.1 ** (admm_epoch_offset//admm_step))

    for param_group in optimizer.param_groups:
         param_group['lr'] = lr

def zero_masking(config):
    masks = {}
    for name,W in config.model.named_parameters():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)
        if name in config.prune_ratios:
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)            
            masks[name] = torch.from_numpy(indices).cuda()
    config.zero_masks = masks
def masking(config):
    masks = {}
    for name,W in config.model.named_parameters():
        if name in config.prune_ratios:           
            above_threshold, pruned_weight = weight_pruning(config,W,config.prune_ratios[name])
            W.data = pruned_weight
            masks[name] = above_threshold
            
    config.masks = masks

