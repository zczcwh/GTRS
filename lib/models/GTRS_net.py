import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import PAM, MRM


class GTRS(nn.Module):
    def __init__(self, num_joint, graph_adj):
        super(GTRS, self).__init__()

        self.num_joint = num_joint
        self.pose_PAM = PAM.get_model(num_joint, embed_dim=128, graph_adj=graph_adj,
                                                  pretrained=cfg.MODEL.posenet_pretrained)

        self.pose_MRM = MRM.get_model(num_joint=num_joint, embed_dim=128, depth=4)  ### output 6890x3


    def forward(self, pose2d):
        pose3d, x_features = self.pose_PAM(pose2d.view(len(pose2d), -1))
        pose3d = pose3d.reshape(-1, self.num_joint, 3)
        cam_mesh = self.pose_MRM(x_features)

        return cam_mesh, pose3d


def get_model(num_joint, graph_adj):
    model = GTRS(num_joint, graph_adj)

    return model


