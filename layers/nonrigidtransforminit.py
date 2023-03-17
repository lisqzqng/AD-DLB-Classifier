import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

class NonRigidTransformInit(nn.Module):
    def __init__(self, num_frames, num_joints):
        super().__init__()
        self.num_frames, self.joints = num_frames, num_joints
        self.num_seq = 100 #new generated sequences for averaging

        angles = torch.Tensor(self.num_frames*self.joints,3)
        nn.init.uniform_(angles, a=-0.3,b=0.3)
        angles = tgmc.conversions.angle_axis_to_rotation_matrix(angles)[:,:3,:3]
        self.weights = nn.Parameter(angles)

    def forward(self, x):
        w_times_x= torch.matmul(x,self.weights)
        return w_times_x
