import torch.nn as nn
import numpy as np
import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

# class MGCN_randn(nn.Module):
#     """
#     Semantic graph convolution layer
#     code is from 'Modulated Graph Convolutional Network for 3D Human Pose Estimation'
#     https://github.com/ZhimingZo/Modulated-GCN
#     """
#     #####  if adj is none, adj is initial as randn
#     def __init__(self, in_features, out_features, in_channel, adj,
#                  fix = False, bias=True):
#         super(MGCN_randn, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         if adj == None:
#             adj = nn.Parameter(torch.randn(in_channel, in_channel, dtype=torch.float))
#
#         if fix == False:
#             self.adj = nn.Parameter(adj, requires_grad = True)
#         else:
#             self.adj = nn.Parameter(adj, requires_grad = False)
#
#         self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#         self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
#         nn.init.xavier_uniform_(self.M.data, gain=1.414)
#
#         self.adj2 = nn.Parameter(torch.ones_like(adj))
#         nn.init.constant_(self.adj2, 1e-6)
#
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
#             stdv = 1. / math.sqrt(self.W.size(2))
#             self.bias.data.uniform_(-stdv, stdv)
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, input):
#         h0 = torch.matmul(input, self.W[0])
#         h1 = torch.matmul(input, self.W[1])
#
#         adj = self.adj.to(input.device) + self.adj2.to(input.device)
#         adj = (adj.T + adj) / 2
#         E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
#
#         output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
#         if self.bias is not None:
#             return output + self.bias.view(1, 1, -1)
#         else:
#             return output



class MGCN(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, in_channel, adj, fix = False, bias=True):
        super(MGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if adj == None:
            adj = nn.Parameter(torch.ones(in_channel, in_channel, dtype=torch.float, requires_grad = True))

        if fix == True:
            self.adj = nn.Parameter(adj, requires_grad = False)
        else:
            self.adj = nn.Parameter(adj, requires_grad = True)

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj) / 2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

class MGCN(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, in_channel, adj, fix = False, bias=True):
        super(MGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if adj == None:
            adj = nn.Parameter(torch.ones(in_channel, in_channel, dtype=torch.float, requires_grad = True))

        if fix == True:
            self.adj = nn.Parameter(adj, requires_grad = True)
        else:
            self.adj = nn.Parameter(adj, requires_grad = False)

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj) / 2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

class Conv(nn.Module):
    def __init__(self, in_channel, drop=0.):
        super().__init__()

        self.channel = in_channel
        self.conv1 = nn.Conv1d(in_channel, 32, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(in_channel, 32, kernel_size=3, padding= 1)
        self.conv3 = nn.Conv1d(32, in_channel, kernel_size=3, padding=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 =self.conv1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2 = self.conv2(x)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        x = x1 + x2
        x = self.conv3(x)
        return x

class MLP_SE(nn.Module):
    def __init__(self, in_features, in_channel, hidden_features=None):
        super().__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_down1 = nn.Linear(in_features*in_channel, in_channel)
        self.fc_down2 = nn.Linear(in_channel, 2*in_channel)
        self.fc_down3 = nn.Linear(2*in_channel, in_channel)
        self.sigmoid = nn.Sigmoid()

        self.act = nn.GELU()

    def forward(self, x):
        B = x.shape[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        ####up_stream
        x1 = x
        ### down_stream
        x2 = x.view(B,-1)
        x2 = self.fc_down1(x2).view(B,1,-1)
        x2 = self.act(x2)
        x2 = self.fc_down2(x2)
        x2 = self.act(x2)
        x2 = self.fc_down3(x2)
        x2 = self.sigmoid(x2)
        #### out
        x = ((x1.transpose(1,2))*x2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x