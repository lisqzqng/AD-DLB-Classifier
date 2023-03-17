from layers.rigidtransform import RigidTransform
from layers.nonrigidtransform import NonRigidTransform
from layers.rigidtransforminit import RigidTransformInit
from layers.nonrigidtransforminit import NonRigidTransformInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc


depth_1 = 128
kernel_size_1 = 3
stride_size = 2
depth_2 = 64
kernel_size_2 = 1
num_hidden = 512
num_labels = 60
dims = 3

class RigidNet60(nn.Module):
    def __init__(self, mod = 'RigidTransform', num_frames = 100, num_joints = 25):
        super(RigidNet60, self).__init__()
        self.num_channels = num_joints * dims
        self.mod = mod
        self.num_frames = num_frames
        self.num_joints = num_joints
        if mod == 'RigidTransform':
            self.rot = RigidTransform(num_frames,num_joints)
        elif mod == 'RigidTransformInit':
            self.rot = RigidTransformInit(num_frames,num_joints)
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.num_channels, depth_1,kernel_size=kernel_size_1, stride=stride_size),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(depth_1, depth_2, kernel_size=kernel_size_2, stride=stride_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.LSTM = nn.LSTM(12, hidden_size=12, bidirectional =True)  
        self.pool=nn.MaxPool1d(kernel_size=2, stride=stride_size)
        self.fc1 = nn.Sequential(
            nn.Linear(depth_2*24, num_hidden),
            #nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            #nn.ReLU(),
            #nn.Dropout(0.5)
        )
        
    def forward(self, x):
        x = self.rot(x)
        x = x.view(x.size(0),self.num_joints*dims,self.num_frames)
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x,  _= self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
