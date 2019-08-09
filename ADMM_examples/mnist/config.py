'''configuration function:
'''
import yaml
import torch.nn as nn

class Config:
    def __init__(self, args):
        """
        read config file
        """
        config_dir = args.config_file
        stage_choices = ['admm','pretrain','retrain']
        stage = args.stage
        if stage not in stage_choices:
            raise Exception("unkown stage.valid choices are {}".format(str(stage_choices)))
        try:
            with open(config_dir, "r") as stream:
                raw_dict = yaml.load(stream)

                # adv parameters
                self.epsilon = raw_dict['adv']['epsilon']
                self.num_steps = raw_dict['adv']['num_steps']
                self.step_size = raw_dict['adv']['step_size']
                self.random_start = raw_dict['adv']['random_start']
                self.loss_func = raw_dict['adv']['loss_func']
                self.width_multiplier = raw_dict['adv']['width_multiplier']
                self.init_func = raw_dict['adv']['init_func']
                self.random_seed = raw_dict['adv']['random_seed']
                # general
                self.print_freq = raw_dict['general']['print_freq']
                self.resume = raw_dict['general']['resume']
                self.gpu = raw_dict['general']['gpu_id']                
                self.arch = raw_dict['general']['arch']
                self.workers = raw_dict['general']['workers']
                self.logging = raw_dict['general']['logging']
                self.log_dir = raw_dict['general']['log_dir']
                try:
                    self.smooth_eps = raw_dict['general']['smooth_eps']
                    self.alpxha = raw_dict['general']['alpha']
                except:
                    print ('no fancy stuff for mnist')
                self.sparsity_type = raw_dict['general']['sparsity_type']                
                self._prune_ratios = raw_dict[self.arch]['prune_ratios']
                self.name_encoder = {}
                self.lr = float(raw_dict[stage]['lr'])
                self.lr_scheduler = raw_dict[stage]['lr_scheduler']             
                self.optimizer = raw_dict[stage]['optimizer']                
                self.save_model = raw_dict[stage]['save_model']
                self.load_model = None  # otherwise key error
                self.masked_progressive = None 
                if stage !='pretrain':
                    self.load_model = raw_dict[stage]['load_model']
                    self.masked_progressive = raw_dict[stage]['masked_progressive']
                if stage =='pretrain' and 'load_model' in raw_dict[stage]:
                    self.load_model = raw_dict[stage]['load_model']
                self.epochs = raw_dict[stage]['epochs']

                try:
                    if stage !='admm':
                        self.warmup_epochs = raw_dict[stage]['warmup_epochs']
                        self.warmup_lr = raw_dict[stage]['warmup_lr']
                except:
                    print ('no fancy stuff for mnist')
                self.admm = (stage == 'admm')
                self.masked_retrain = (stage =='retrain')
                self.rho = None
                self.rhos = {}
                if stage == 'admm':
                # admm_pruning
                    self.admm_epoch = raw_dict[stage]['admm_epoch']
                    self.rho = raw_dict[stage]['rho']                
                    self.multi_rho = raw_dict[stage]['multi_rho']
                    self.verbose = raw_dict[stage]['verbose']
                # following variables assist the pruning algorithm

                self.masks = None
                self.zero_masks = None
                self.conv_names = []
                self.bn_names = []
                self.fc_names = []
                self.prune_ratios = {}
                self.model = None

        except yaml.YAMLError as exc:
            print(exc)
    def prepare_pruning(self):
         print ('inside prepare')
         self._extract_layer_names(self.model)
         for good_name,ratio in self._prune_ratios.items():
             self._encode(good_name)
         for good_name,ratio in self._prune_ratios.items():
             self.prune_ratios[self.name_encoder[good_name]] = ratio
         for k in self.prune_ratios.keys():
             self.rhos[k] = self.rho  # this version we assume all rhos are equal                                 
    def __str__(self):
        return str(self.__dict__)
    def _extract_layer_names(self,model):
         """
         Store layer name of different types in arrays for indexing
         """
         
         names = []
         for name, W in self.model.named_modules():
             names.append(name)
         print (names)
         for name,W in self.model.named_modules():             
             name+='.weight'  # name in named_modules looks like module.features.0. We add .weight into it
             if isinstance(W,nn.Conv2d):
                 self.conv_names.append(name)
             if isinstance(W,nn.BatchNorm2d):
                 self.bn_names.append(name)
             if isinstance(W,nn.Linear):
                 self.fc_names.append(name)
    def _encode(self,name):
         """
         Examples:
         conv1.weight -> conv           1                weight
                         conv1-> prefix   weight->postfix        
                         conv->layer_type  1-> layer_id + 1  weight-> postfix
         Use buffer for efficient look up  
         """
         prefix,postfix = name.split('.')
         dot_position = prefix.find('.')
         layer_id = ''
         for s in prefix:
             if s.isdigit():
                 layer_id+=s
         id_length = len(layer_id)         
         layer_type = prefix[:-id_length]
         layer_id = int(layer_id)-1
         if layer_type =='conv' and len(self.conv_names)!=0:
             self.name_encoder[name] = self.conv_names[layer_id]
         elif layer_type =='fc' and len(self.fc_names)!=0:
             self.name_encoder[name] =  self.fc_names[layer_id]
         elif layer_type =='bn' and len(self.bn_names)!=0:             
             self.name_encoder[name] =  self.bn_names[layer_id]             
