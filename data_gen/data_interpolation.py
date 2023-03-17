import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
import math
import time
skel_path = '../data/nturgb_d/xsub/'

#Must have data in shape: N_samples,N_frames,N_joints,N_dims(=3)


train_file_name = 'body1_train_xsub_raw100.npy' 
test_file_name = 'body1_test_xsub_raw100.npy' 
train_label_name = 'train_label.pkl'
test_label_name = 'test_label.pkl'

num_frames = 100

print('____________________Loading Train Data____________________')
X_train= np.load(os.path.join(skel_path,train_file_name), allow_pickle=True)
new_X_train = np.zeros((X_train.shape[0],num_frames,X_train.shape[2],X_train.shape[3]))
print('____________________Loaded Train Data_____________________')
count = 0
print('_____________Proceeding to preprocess Train data__________')
##START HERE
## Number_of_samples * number_of_frames * joints * dimensions
start = time.time()
for x in X_train:
    new_seq = list()
    zero_row = []
    for i in range(len(x)):
        if (x[i, :] == np.zeros((1, 25,3))).all():
            zero_row.append(i)
    new_seq = np.delete(x, zero_row, axis=0)
    seq_len = new_seq.shape[0]

    if np.any(new_seq) == False or seq_len < 30:
        #print('skipped skeleton n°{}, seq_len: {}'.format(count,seq_len))
        new_X_train[count] = np.zeros((num_frames,25,3))
        count+=1
        continue
    #print(seq_len)
    interp_seq = list()
    for joint in range(x.shape[1]):
        new_axis = list()
        fx = new_seq[:,joint,0] # axis
        cs = CubicSpline(np.arange(seq_len),fx)
        interpx = cs(np.arange(0,seq_len,seq_len/num_frames))
        fy = new_seq[:,joint,1]
        cs = CubicSpline(np.arange(seq_len),fy)
        interpy = cs(np.arange(0,seq_len,seq_len/num_frames))
        fz = new_seq[:,joint,2]
        cs = CubicSpline(np.arange(seq_len),fz)
        interpz = cs(np.arange(0,seq_len,seq_len/num_frames))
        new_joint = np.vstack((interpx,interpy,interpz))
        interp_seq.append(new_joint)
    interp_seq = np.array(interp_seq).transpose(2,0,1)[0:num_frames]
    new_X_train[count] = interp_seq
    count+=1
print('_______ train data shape {} ________'.format(new_X_train.shape))
print(new_X_train.shape)
end = time.time()
print('Time taken to interp : {}'.format((end-start) / len(X_train)))
##END HERE
print('___________________ Saving Train Data ____________________')
np.save(os.path.join(skel_path,'train_xsub_interp_{}_frames.npy'.format(num_frames)),new_X_train, allow_pickle= True)
#free memory
X_train = None
new_X_train = None
print('_________________Saving train data done___________________')
print('___________________Loading Test data______________________')
X_test = np.load(os.path.join(skel_path,test_file_name), allow_pickle=True)
new_X_test = np.zeros((X_test.shape[0],num_frames,X_test.shape[2],X_test.shape[3]))
print('___________________Loaded Test data_______________________')
count = 0
print('_____________Proceeding to preprocess test data__________')
for x in X_test:
    new_seq = list()
    zero_row = []
    for i in range(len(x)):
        if (x[i, :] == np.zeros((1, 25,3))).all():
            zero_row.append(i)
    new_seq = np.delete(x, zero_row, axis=0)
    seq_len = new_seq.shape[0]
    if np.any(new_seq) == False or seq_len < 30:
        #print('skipped skeleton n°{}, seq_len: {}'.format(count,seq_len))
        new_X_test[count] = np.zeros((num_frames,25,3))
        count+=1
        continue
    #print(seq_len)
    interp_seq = list()
    for joint in range(x.shape[1]):
        new_axis = list()
        fx = new_seq[:,joint,0]
        cs = CubicSpline(np.arange(seq_len),fx)
        interpx = cs(np.arange(0,seq_len,seq_len/num_frames))
        fy = new_seq[:,joint,1]
        cs = CubicSpline(np.arange(seq_len),fy)
        interpy = cs(np.arange(0,seq_len,seq_len/num_frames))
        fz = new_seq[:,joint,2]
        cs = CubicSpline(np.arange(seq_len),fz)
        interpz = cs(np.arange(0,seq_len,seq_len/num_frames))
        new_joint = np.vstack((interpx,interpy,interpz))
        interp_seq.append(new_joint)
    interp_seq = np.array(interp_seq).transpose(2,0,1)[0:num_frames]
    new_X_test[count] = interp_seq
    count+=1
print('________Test data shape {} ________'.format(new_X_test.shape))
print(new_X_test.shape)

print('___________________ Saving Test Data ___________________')
np.save(os.path.join(skel_path,'test_xsub_interp_{}_frames.npy'.format(num_frames)),new_X_test, allow_pickle= True)
X_test = None
new_X_test = None
print('__________________Saving test data done__________________')