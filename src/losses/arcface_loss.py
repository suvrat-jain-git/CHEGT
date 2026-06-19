"""ArcFace Loss."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.5, scale=64.0, easy_margin=False):
        super().__init__()
        self.margin      = margin
        self.scale       = scale
        self.easy_margin = easy_margin
        self.weight      = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        emb_norm    = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight,  p=2, dim=1)
        cosine = (emb_norm @ weight_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine   = torch.sqrt(1.0 - cosine ** 2)
        phi    = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(output, labels.long())
