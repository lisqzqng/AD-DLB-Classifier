import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

class RigidTransform(nn.Module):
    def __init__(self, num_frames, num_joints, ):
        super().__init__()
        self.num_frames, self.joints = num_frames, num_joints
        self.num_seq = num_frames #new generated sequences for averaging

        angles = torch.Tensor(self.num_frames,3)
        nn.init.uniform_(angles, a=-0.3,b=0.3) # weight init
        self.weights = nn.Parameter(angles)  # nn.Parameter is a Tensor that's a module parameter.


    def forward(self, x):
        new_weights = tgmc.conversions.angle_axis_to_rotation_matrix(self.weights)[:,:3,:3]
        w_times_x= torch.matmul(x,new_weights)
        return w_times_x
