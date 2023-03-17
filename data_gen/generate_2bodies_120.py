import numpy as np
import os 
import copy 
from scipy.interpolate import CubicSpline
import math
import argparse

train_data_file = 'train_data_joint_pad.npy'
test_data_file = 'val_data_joint_pad.npy'


parser = argparse.ArgumentParser()

parser.add_argument('--prot', default='xsub', type=str,help = 'xsub or xview')
parser.add_argument('--save_raw', default=False, type=bool)
parser.add_argument('--save_one_body', default=False, type=bool)
parser.add_argument('--save_50x3',default=False,type=bool,help= 'Save in format: N_samples*N_frames*50x3')
parser.add_argument('--num_frames',default=100,type=int,help='Number of frames to consider from source data')
parser.add_argument('--num_frames_interp',default=100,type=int,help='Target Number of frames')
opt = parser.parse_args()

save_raw = opt.save_raw
save_one_body = opt.save_one_body
save_joints_sep = opt.save_50x3

num_frames = opt.num_frames
interp_frames = opt.num_frames_interp

prot = opt.prot

if prot not in ['xsub','xsetup']:
    print('Invalid protocol for NTU60 will be using the cross_subject protocol')
    prot = 'xsub'
path = '../data/nturgb_d120/{}/'.format(prot)


logmap = False #this one is useless but keep it as is. there was an old logmap code but didn't work, we'll update this file soon.
#we plan to add the inverse_exp files back to this script
#as well as parallel transport
lm = '_invexp' if logmap == True else ''


do_train = True #deactivate in case of fail of either
do_test = True


def interp_data(ske_joints,interp_frames = 30, min_seq_len = 30):
    num_frames = ske_joints.shape[1]
    joints = ske_joints.shape[2]
    dims = ske_joints.shape[3]

    print('____________________Loading Data____________________')
    res = np.zeros((ske_joints.shape[0],interp_frames,joints,dims))
    print('____________________Loaded Data_____________________')
    count = 0
    print('_____________Proceeding to preprocess data__________')
    for x in ske_joints:
        new_seq = list()
        zero_row = []
        for i in range(len(x)):
            if (x[i, :] == np.zeros((1, joints,dims))).all():
                zero_row.append(i)
        new_seq = np.delete(x, zero_row, axis=0)
        seq_len = new_seq.shape[0]
        if np.any(new_seq) == False or seq_len < min_seq_len:
            print('skipped skeleton n°{}, seq_len: {}'.format(count,seq_len))
            res[count] = np.zeros((interp_frames,joints,dims))
            count+=1
            continue
        #print(seq_len)
        interp_seq = list()
        for joint in range(x.shape[1]):
            new_axis = list()
            fx = new_seq[:,joint,0]
            cs = CubicSpline(np.arange(seq_len),fx)
            interpx = cs(np.arange(0,seq_len,seq_len/interp_frames))
            fy = new_seq[:,joint,1]
            cs = CubicSpline(np.arange(seq_len),fy)
            interpy = cs(np.arange(0,seq_len,seq_len/interp_frames))
            fz = new_seq[:,joint,2]
            cs = CubicSpline(np.arange(seq_len),fz)
            interpz = cs(np.arange(0,seq_len,seq_len/interp_frames))
            new_joint = np.vstack((interpx,interpy,interpz))
            interp_seq.append(new_joint)
        interp_seq = np.array(interp_seq).transpose(2,0,1)[0:interp_frames]
        #print(interp_seq.shape)
        res[count] = interp_seq
        count+=1
    print('_______ data shape {} ________'.format(res.shape))
    print(res.shape)
    return res

if do_train == True:
    train_data = np.load(os.path.join(path,train_data_file),allow_pickle=True)
    print(train_data.shape)

    print('Destacking bodies before interpolation')
    train_data = train_data.transpose([0,4,2,3,1])
    train_data = train_data[:,:,:num_frames,:,:]

    body_1 = train_data[:,0,:,:,:]
    body_2 = train_data[:,1,:,:,:]

    joints = body_1.shape[2]
    dims = body_1.shape[3]

    train_data = None
    if save_raw == True:
        np.save(os.path.join(path,'body1_train_{}_raw100.npy'.format(prot)),body_1)
        np.save(os.path.join(path,'body2_train_{}_raw100.npy'.format(prot)),body_2)

    print('interpolating each body [train] before stacking')
    body_1 = interp_data(body_1, interp_frames = interp_frames)
    body_2 = interp_data(body_2, interp_frames = interp_frames)

    if save_one_body == True:
        print('saving interp bodies [train] before stacking')
        np.save(os.path.join(path,'body1_train_{}_interp100.npy'.format(prot)),body_1)
        np.save(os.path.join(path,'body2_train_{}_interp100.npy'.format(prot)),body_2)

    if save_joints_sep == True:
        new_train_data_stacked = np.concatenate((body_1,body_2),axis = 2)
        print('Stacked 50x3 data shape: {}'.format(new_train_data_stacked.shape))
        np.save(os.path.join(path,'train_{}_interp100_50x3.npy'.format(prot)),new_train_data_stacked)

    body_1 = body_1.reshape(body_1.shape[0],interp_frames,joints*dims)
    body_2 = body_2.reshape(body_2.shape[0],interp_frames,joints*dims)

    new_train_data = np.concatenate((body_1,body_2),axis = 2)
    print('Stacked Frames*150 data shape: {}'.format(new_train_data.shape))
    body_1 = None
    body_2 = None #empty ram

    np.save(os.path.join(path,'train{}_{}_interp{}.npy'.format(lm,prot,interp_frames)),new_train_data)
    new_train_data = None
    print('Train data preprocessing done')

if do_test == True:
    test_data = np.load(os.path.join(path,test_data_file),allow_pickle=True)
    print(test_data.shape)

    print('Destacking bodies before interpolation')
    test_data = test_data.transpose([0,4,2,3,1])
    test_data = test_data[:,:,:num_frames,:,:]

    body_1 = test_data[:,0,:,:,:]
    body_2 = test_data[:,1,:,:,:]
    
    joints = body_1.shape[2]
    dims = body_1.shape[3]
    test_data = None

    if save_raw == True:
        np.save(os.path.join(path,'body1_test_{}_raw100.npy'.format(prot)),body_1)
        np.save(os.path.join(path,'body2_test_{}_raw100.npy'.format(prot)),body_2)

    print('interpolating each body [test] before stacking')
    body_1 = interp_data(body_1, interp_frames = interp_frames)
    body_2 = interp_data(body_2, interp_frames = interp_frames)

    if save_one_body == True:
        print('saving interp bodies [test] before stacking')
        np.save(os.path.join(path,'body1_test_{}_interp100.npy'.format(prot)),body_1)
        np.save(os.path.join(path,'body2_test_{}_interp100.npy'.format(prot)),body_2)
        
    if save_joints_sep == True:
        new_test_data_stacked = np.concatenate((body_1,body_2),axis = 2)
        print('Stacked 50x3 data shape: {}'.format(new_test_data_stacked.shape))
        np.save(os.path.join(path,'test_{}_interp100_50x3.npy'.format(prot)),new_test_data_stacked)

    body_1 = body_1.reshape(body_1.shape[0],interp_frames,joints*dims)
    body_2 = body_2.reshape(body_2.shape[0],interp_frames,joints*dims)

    new_test_data = np.concatenate((body_1,body_2),axis = 2)
    print(new_test_data.shape)

    body_1 = None
    body_2 = None #empty ram


    np.save(os.path.join(path,'test{}_{}_interp{}.npy'.format(lm,prot,interp_frames)),new_test_data)
    new_test_data = None
    print('Test data preprocessing done')
