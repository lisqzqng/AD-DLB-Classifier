import os
import os.path as osp
import numpy as np
from numpy import ndarray
import pickle
import random
from tqdm import tqdm
import copy
from processing.CenteredScaled import CenteredScaled
from processing.inv_exp import inv_exp

import joblib
import pandas as pd

BASE_DIR = "/home/dw/Documents/DAMoS/Code/VIBE/data/vpare_db/skeletons/5-fold/"
updrs_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/datasets/hospital/annotations/updrs.csv"
diag_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/datasets/hospital/annotations/diag.csv"
## Hyperparameters
SEQLEN = 50
MIN_REST = 15
save_fp_format = './skeletons/5-fold/{:s}*{:s}.pkl' # train or test

## Load the annotations into dictionary
annos_diag = pd.read_csv(diag_fp, header=None).to_numpy()
annos_updrs = pd.read_csv(updrs_fp, header=None).to_numpy()
diag_dict = {}
updrs_dict = {}
assert annos_diag.shape[0]==annos_updrs.shape[0]
for i in range(annos_diag.shape[0]):
    diag_dict[annos_diag[i, 0].split('.')[0]] = annos_diag[i, 1]
    updrs_dict[annos_updrs[i, 0].split('.')[0]] = annos_updrs[i, 1]

for sklt_name in tqdm(sorted(os.listdir(BASE_DIR))):
    jsonpath = osp.join(BASE_DIR, sklt_name)
    info = joblib.load(jsonpath)

    ## Split the whole sequence into chunks of 50 frames
    All_train, All_test = [], []
    # Process video by video
    # ref_skel = copy.deepcopy(data[0]) # before all space transformations
    for k,v in tqdm(info.items()):
        d={}
        name = osp.basename(k).split('.')[0]
        # d['vid_name']=name
        d['num_frames']=SEQLEN
        d['diag']=diag_dict[name]
        d['gait_score']=updrs_dict[name]
        ## preprocessing (going to preshape space)
        data = np.array(v)
        ## preprocess indices
        last_frame = data.shape[0]-1
        if last_frame < SEQLEN - 11:
            print(f"Video {name} has only {last_frame+1} frames !!")
            continue
        elif last_frame < SEQLEN-1:
            print(f"Video {name} has only {last_frame+1} frames.")
            data = np.concatenate([data, np.repeat(data[-1:], SEQLEN-last_frame-1, axis=0)], axis=0)
            last_frame = SEQLEN-1
        ## split data into train splits
        index_train = np.arange(0, last_frame-SEQLEN//2, SEQLEN//2)
        if last_frame - index_train[-1] < SEQLEN-1:
            index_train = index_train[:-1]
        if last_frame - index_train[-1] > MIN_REST:
            index_train = np.append(index_train, last_frame-SEQLEN)
        ## split the data into test splits
        index_test = np.arange(0, last_frame, SEQLEN)
        if last_frame - index_test[-1] < SEQLEN-1:
            index_test = index_test[:-1]
        for idx, c in enumerate(index_train):
            start, end = c, c+SEQLEN
            cdata = copy.deepcopy(data[start:end])
            ## by default, take cast=='log_sref'
            for i in range(cdata.shape[0]):
                try:
                    cdata[i] = CenteredScaled(cdata[i])
                    if i==0:
                        ref_skel = copy.deepcopy(cdata[0])
                    ## From the pre-shape space to a tangent space...
                    cdata[i] = inv_exp(ref_skel, cdata[i])
                except:
                    raise ValueError('!')
            d['data'] = cdata
            d['vid_name'] = name+f'*{idx}'
            All_train.append(d)
        for idx, c in enumerate(index_test):
            start, end = c, c+SEQLEN
            cdata = copy.deepcopy(data[start:end])
            ## by default, take cast=='log_sref'
            for i in range(cdata.shape[0]):
                try:
                    cdata[i] = CenteredScaled(cdata[i])
                    if i==0:
                        ref_skel = copy.deepcopy(cdata[0])
                    ## From the pre-shape space to a tangent space...
                    cdata[i] = inv_exp(ref_skel, cdata[i])
                except:
                    raise ValueError('!')
            d['data'] = cdata
            d['vid_name'] = name+f'*{idx}'
            All_test.append(d)
        
    # dump the data
    fp_name = osp.basename(sklt_name).split('.')[0].split('2023_')[-1]
    save_fp_train = save_fp_format.format(fp_name, 'train')
    save_fp_test = save_fp_format.format(fp_name, 'test')
    os.makedirs(osp.dirname(save_fp_train), exist_ok=True)
    with open(save_fp_train, 'wb') as f:
        pickle.dump(All_train, f)
    with open(save_fp_test, 'wb') as f:
        pickle.dump(All_test, f)
