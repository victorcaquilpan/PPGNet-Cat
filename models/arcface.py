# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F



# Creation of ArcFace Loss
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s = 64, m=0.3):
        super().__init__()
        self.feature_in = in_features
        self.feature_out = out_features
        self.scale = s
        self.margin = m 
        
        self.weights = nn.Parameter(torch.FloatTensor(self.feature_out,self.feature_in))
        nn.init.xavier_normal_(self.weights)
    def forward(self, features, targets):
        cos_theta = F.linear(features, F.normalize(self.weights), bias=None)
        cos_theta = cos_theta.clip(-1+1e-7, 1-1e-7)
        
        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes = self.feature_out) * self.margin
        arc_cos += M
        
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return logits