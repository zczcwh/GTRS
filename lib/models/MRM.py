from core.config import cfg
import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.backbones.net import Attention, Conv, MLP_SE

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
# SMPL_MEAN_PARAMS = osp.join(BASE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MEAN_vertices = osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')



class Block(nn.Module):

    def __init__(self, dim, in_channel, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.conv = Conv(in_channel=in_channel, drop=drop)
        self.mlpse = MLP_SE(in_features = dim, in_channel=in_channel,hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)) + self.conv(self.norm1(x)))
        x = x + self.drop_path(self.mlpse(self.norm2(x)))
        return x

class Cross_Block(nn.Module):

    def __init__(self, dim, pose_channel, temp_channel, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.pose_channel = pose_channel
        self.Pose_Block = Block(
            dim=dim , in_channel=pose_channel, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Temp_Block = Block(
            dim=dim, in_channel=temp_channel, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)

        self.Fuse_Block = Block(
            dim=dim, in_channel=temp_channel + pose_channel, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer)


    def forward(self, x):
        x_pose = x[:,:self.pose_channel,:]
        x_temp = x[:, self.pose_channel:, :]
        x_pose = self.Pose_Block(x_pose)
        x_temp = self.Temp_Block(x_temp)
        x = torch.cat((x_pose, x_temp), dim = 1)
        x = self.Fuse_Block(x)
        return x

class Mesh_Regression(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, num_joint=17, embed_dim=128, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, act_layer=None):
        super().__init__()
        self.patch_number = num_joint

        act_layer = act_layer or nn.GELU
        self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Cross_Block(
            dim=self.embed_dim , pose_channel=self.patch_number,temp_channel=15, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).unsqueeze(0)
        self.register_buffer('init_vertices', init_vertices)

        self.up_feature = nn.Linear(embed_dim, embed_dim)
        self.up_linear = nn.Linear(689*2, embed_dim)

        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(self.embed_dim )

        self.d_conv = nn.Conv1d(self.patch_number + 15, 3, kernel_size=3, padding=1)
        self.d_linear = nn.Linear(self.embed_dim , 6890)


    def forward(self, features_3d):
        ###### input features_3d [B,J, emb_dim]
        B = features_3d.shape[0]
        features_3d = features_3d.view(B, self.patch_number, -1)
        features_3d = self.up_feature(features_3d)

        init_vertices = self.init_vertices.expand(B, -1, -1)
        mean_smpl = self.up_linear(init_vertices.transpose(1,2).reshape(B,15,-1))

        x = torch.cat((features_3d, mean_smpl), dim=1)
        x = self.blocks(x)
        x = self.norm(x)

        ####### after mesh regression module, x_out [B, J+T, emb_dim]
        x_out = x
        x_out = self.d_linear(x_out)
        x_out = self.gelu(x_out)
        x_out = self.d_conv(x_out).transpose(1, 2)
        x_out = x_out + init_vertices

        return x_out


def get_model(num_joint=17, embed_dim=64, depth=4):
    model = Mesh_Regression(num_joint, embed_dim, depth)

    return model

