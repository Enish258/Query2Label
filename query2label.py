import os, sys
import os.path as osp

import torch
import torch.nn as nn

import numpy as np
import math

from src.models.query2label.backbone import build_backbone
from src.models.query2label.transformer import build_transformer


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class query2label(nn.Module):
    def __init__(self, backbone, transfomer, num_class,backbone_name):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        self.backbone_name=backbone_name

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)


    def forward(self, input):
        src, pos = self.backbone(input)
        if self.backbone_name=='swin':
            src = torch.permute(src[0], (0,3, 1, 2)) #because output from swin is in order (Batch_size,H,W,Channels)
            pos = pos[-1]
        else:
            src, pos = src[-1], pos[-1]
       
        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, self.input_proj(pos))[0] # B,K,d
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        return out


def build_q2l(cfg):

    backbone = build_backbone(cfg)
    transformer = build_transformer()

    model = query2label(
        backbone = backbone,
        transfomer = transformer,
        num_class = cfg['model']['params']['num_classes'],
        backbone_name = cfg['model']['params']['backbone_type']
    )
    

    return model
        
