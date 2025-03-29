import random
import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE
import numpy as np
from functional import drop_edge_weighted_new, pr_drop_weights_new, normalize_edge_weights


class ACA_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(ACA_GCN, self).__init__()
        K = 3
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
                torch.nn.Linear(cls_input_dim, 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                torch.nn.Linear(256, num_classes))
        self.edge_net = PAE(input_dim=edgenet_input_dim // 2, dropout=dropout)
        self.model_init


    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, pd_ftr_dim, nonimg, m, flag , enforce_edropout=False):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.edge_dropout > 0:
            if enforce_edropout or self.training:

                one_mask = torch.ones([edgenet_input.shape[0], 1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))


                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask]

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        if flag == 1:#EWP
            p=0.5
            pr_weights = pr_drop_weights_new(edge_index, aggr='sink', k=10).to(device)
            edge_weight_norm = normalize_edge_weights(edge_weight)
            selsect_weight = p*edge_weight_norm + (1-p)*pr_weights
            # selected = edge_weight[selsect_weight < 0.85]
            selected = edge_weight[edge_weight < 0.85]  # 0.85
            n = len(selected)
            # idx = torch.nonzero(selsect_weight < 0.65).squeeze()
            idx = torch.nonzero(edge_weight < 0.85).squeeze()
            # 确保 idx 是一维张量
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            for i in range(n):
                j = random.randint(i, n - 1)
                idx_i, idx_j = idx[i].item(), idx[j].item()
                edge_weight[idx_i], edge_weight[idx_j] = edge_weight[idx_j], edge_weight[idx_i]


        elif flag == 2:#EP
            pr_weights = pr_drop_weights_new(edge_index, aggr='sink', k=10).to(device)
            def drop_edge():
                return drop_edge_weighted_new(edge_index, edge_weight, pr_weights, edgenet_input, p=0.2, threshold=0.6)
            edge_index, edgenet_input = drop_edge()

            edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index)
        features = F.dropout(features, self.dropout, self.training)
        # print("Type of edge_index at the start of forward:", type(edge_index))
        edge_index = torch.tensor(edge_index, dtype=torch.float32).to(device) if isinstance(edge_index, np.ndarray) else edge_index.to(device)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.cls(jk)

        return h0, logit


    #loss2
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # return -torch.log(between_sim.diag() / between_sim.sum(1))

    def loss(self, h1: torch.Tensor, h2: torch.Tensor, mean: bool = True):
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
