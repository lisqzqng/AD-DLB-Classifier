import torch
import math
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc
import copy
from processing.CenteredScaled import CenteredScaled
from processing.inv_exp import inv_exp
from models.RigidNet60 import RigidNet60
from models.NonRigidNet60 import NonRigidNet60


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='', type=str, help='Path to the .pth checkpoint model file. The model must take NxFxJxD or NxFxJD data shapes')
parser.add_argument('--prot', default='xsub', type=str,help = 'xsub or xview')
parser.add_argument('--batch_size_test', default = 256, type=int)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--data_path', default='', type=str, help= '.npy test file')
parser.add_argument('--labels_path',default='',type=str, help= '.pkl labels file')
opt = parser.parse_args()

use_cuda = opt.use_cuda
#argument block#

cuda = torch.cuda.is_available()
device = 'cuda' if (cuda == True and use_cuda == True) else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()


#default Model and test data path
path = './models/checkpoints/model_frames_NTU60.pth'
data_path = 'test_xsub_interp100.npy'


dir = './data/nturgb_d/xsub/'
file_test = 'test_xsub_interp100.npy'
X_test = np.load(os.path.join(dir,file_test), allow_pickle=True)
y_test = np.load(os.path.join(dir,'val_label.pkl'), allow_pickle=True)
y_test = y_test[1]
y_test = np.array(y_test).astype('int32')

one_body = False #will be changed base on filename if it's stacked or not
if ('body1' in file_test) or ('body2' in file_test):
    one_body = True

if one_body == False:
    print('reshaping (Destacking Joints and dims)')
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]//3 , 3))

if opt.checkpoint != '':
    path = opt.checkpoint

if opt.data_path != '' and opt.label_path != '':
    X_test = np.load(data_path, allow_pickle=True)
    y_test = np.load(label_path, allow_pickle=True)
    y_test = y_test[1]
    y_test = np.array(y_test).astype('int32')
else:
    print('Test path or Labels path (both must be specified) not specified will be testing a default test file for Cross subject protocol')

print('Preprocessing Test data (going to preshape space)')
for i in range(X_test.shape[0]):
    for j in range(X_test[i].shape[0]):
        X_test[i, j] = CenteredScaled(X_test[i, j])

for i in range(X_test.shape[0]):
    for j in range(X_test[i].shape[0]):
        try:
            X_test[i, j] = inv_exp(X_test[i, 0], X_test[i, j])
        except:
            i = i + 1

batch_size_test = opt.batch_size_test


model = torch.load(path)

#Model Testing

correct_test = 0
total_test = 0


with torch.no_grad():
    steps = int(len(X_test) / batch_size_test)
    for i in range(steps):                        
        x, y = X_test[i*batch_size_test:(i*batch_size_test)+batch_size_test], y_test[i*batch_size_test:(i*batch_size_test)+batch_size_test]                
        inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        outputs = model(inputs.float())
        y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
        _, predicted = torch.max(y_pred_softmax, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels.long()).sum().item()

print('Accuracy of the network : %.3f %%' % (
    100 * correct_test / total_test))