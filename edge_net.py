import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE
import random
import numpy as np

class weight(torch.nn.Module):
    def __init__(self, dropout, edge_dropout, edgenet_input_dim):
        super(weight, self).__init__()
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.edge_net = PAE(input_dim=edgenet_input_dim // 2, dropout=dropout)
        self.edge_net.to('cuda')


    def weight_(self, edge_index, edgenet_input, flag, enforce_edropout=False):
        edgenet_input = edgenet_input.to('cuda')
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0],1]).to('cuda')
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask]

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))


        if flag == 1:
            selected = edge_weight[edge_weight < 0.85]  # 0.85
            n = len(selected)
            idx = torch.nonzero(edge_weight < 0.85).squeeze()
            for i in range(n):
                j = random.randint(i, n - 1)
                idx_i, idx_j = idx[i], idx[j]
                edge_weight[idx_i], edge_weight[idx_j] = edge_weight[idx_j], edge_weight[idx_i]


        if flag == 2:
            mask = edge_weight >= 0.85
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]

        if flag == 3:
            n = len(edge_weight)  # 0.85
            n1 = int(0.3 * n)
            al_idx = np.arange(n)
            idx = np.random.choice(al_idx,n1,replace=False)
            for i in range(n1):
                j = random.randint(i, n1 - 1)
                idx_i, idx_j = idx[i], idx[j]
                edge_weight[idx_i], edge_weight[idx_j] = edge_weight[idx_j], edge_weight[idx_i]
        return edge_weight, edge_index
