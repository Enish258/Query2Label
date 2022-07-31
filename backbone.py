import os
import warnings
from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
import src.models as models_module
from src.models.query2label.position_encoding import build_position_encoding
import src.models as models_module



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                pos.append(self[1](x).to(x.dtype))
        else:
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos





def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    backbone_type = cfg["model"]['params']['backbone_type']
    backbone_class = getattr(models_module, cfg["model"]['params']['backbone'])
    backbone = backbone_class(**cfg["model"]["backbone_params"])
    if backbone_type == 'densenet':
        feature_extractor = torch.nn.Sequential(*list(backbone.basemodel.features.children())[:-1])
    elif backbone_type == 'swin':
        feature_extractor = torch.nn.Sequential(*list(backbone.basemodel.children())[0])
    elif backbone_type == 'resnet' :
        feature_extractor = torch.nn.Sequential(*list(backbone.basemodel.children()[:-2]))
    else :
        raise ValueError('invalid_model')
    model = Joiner(feature_extractor, position_embedding)
    model.num_channels = cfg['model']['params']['hidden_dim']
    return model




