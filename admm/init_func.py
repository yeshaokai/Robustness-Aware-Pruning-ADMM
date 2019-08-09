import torch
import torch.nn as nn
# a colleciton of init functions for conv layers

class  Init_Func():
    def __init__(self,init_type):
        self.init_type = init_type

    def init(self,W):
        if self.init_type  == 'default':
            return torch.nn.init.xavier_uniform_(W)
        elif self.init_type == 'orthogonal':
            return torch.nn.init.orthogonal_(W)
        elif self.init_type == 'uniform':
            return torch.nn.init.uniform_(W)
        elif self.init_type == 'normal':
            return torch.nn.init.normal_(W)
        elif self.init_type == 'constant':
            return torch.nn.init.constant_(W)
        elif self.init_type == 'xavier_normal':
            return torch.nn.init.xavier_normal_(W)
        elif self.init_type == 'xavier_uniform':
            return torch.nn.init.xavier_uniform_(W)
        elif self.init_type == 'kaiming_uniform':
            return torch.nn.init.kaiming_uniform_(W)
        elif self.init_type == 'kaiming_normal':
            return torch.nn.init.kaiming_normal_(W)
        else:
            raise Exception ("unknown initialization method")
