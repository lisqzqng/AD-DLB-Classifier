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

import re

# BASE_DIR = "/home/dw/Documents/DAMoS/Code/VIBE/data/vpare_db/skeletons/10-fold/"
# updrs_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/datasets/hospital/annotations/updrs.csv"
# diag_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/datasets/hospital/annotations/diag.csv"
# label_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/data/label_118.xlsx"
# label_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/data/tulip_label_60.xlsx"
## Hyperparameters JBHI
SEQLEN = 70
STRIDE = 30
MIN_REST = 20
FOLD = 1
## Hyperparameters AMAI2023
# SEQLEN = 100
# STRIDE = 50
# MIN_REST = 50
# FOLD = 10
save_dir = f'./skeletons/{FOLD}-fold-amai/'
os.makedirs(osp.dirname(save_dir), exist_ok=True)
save_fp_format = save_dir + '{:s}*{:s}.pkl' # train or valid
# get_updrs_3cls = lambda x: x if x>3 else 2
# get_diag_3cls = lambda x: x if x<2 else (1 if x==3 else 2)

## Load the annotations into dictionary
# df = pd.read_excel(label_fp, sheet_name='label_info', engine='openpyxl')
# annos_label = pd.DataFrame(df, columns=['vidname', 'diag', 'score']).to_numpy()

# for jsonpath in tqdm([osp.join('skeletons',x) for x in os.listdir('skeletons') if 'kinectv2_657.json' in x]):
for jsonpath in tqdm(['skeletons/wham_skeletons_kinectv2_robtulip_transl.json']):
    info = joblib.load(jsonpath)
    keys = ['vidname', 'score', 'diag', 'isbackward', 'patientid']
    out_dict = {k:[] for k in keys}
    subjects = list(set([x.split('_')[1] for x in info.keys() if x.startswith('vid')]))
    subjects.extend(list(set(['_'.join(x.split('_')[:2]) for x in info.keys() if x.startswith('Subject_')])))
    subjects.extend(list(set([int(x[3:5]) for x in info.keys() if x.startswith('OAW')])))
    for k,v in info.items():
        out_dict['vidname'].append(k)
        out_dict['score'].append(v['gait_score'])
        out_dict['diag'].append(v['diag'])
        out_dict['isbackward'].append(0)
        # find the subject id with subjects
        if k.startswith('OAW'):
            sid = int(k[3:5])
        else:
            sid = k.split('_')[1] if k.startswith('vid') else '_'.join(k.split('_')[:2])
        subj_id = subjects.index(sid)
        out_dict['patientid'].append(subj_id)
    # save to csv
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(jsonpath.replace('.json', '.csv'), index=False)
    ## Split the whole sequence into chunks of 50 frames
    All_train = []
    # All_valid = []
    # Process video by video
    # ref_skel = copy.deepcopy(data[0]) # before all space transformations
    for k,v in tqdm(info.items()):
        d={}
        name = osp.basename(k).split('.')[0]
        name = name.replace('f', '')
        # _name = re.sub(r'_CC\d+', '', name)
        # d['vid_name']=name
        d['num_frames']=SEQLEN
        # try:
        #     # d['diag']=get_diag_3cls(annos_label[np.where(annos_label[:, 0] == name)[0]][0, 1])
        #     d['diag']=annos_label[np.where(annos_label[:, 0] == _name)[0]][0, 1]
        # except IndexError:
        #     print(f"Video {name} is not in the annotation file.")
        #     continue
        d['diag'] = v['diag']
        d['gait_score'] = v['gait_score']
        # d['gait_score']=annos_label[np.where(annos_label[:, 0] == _name)[0]][0, 2]
        ## preprocessing (going to preshape space)
        data = v['joints3D']
        ## preprocess indices
        last_frame = data.shape[0]-1
        if last_frame < SEQLEN - 6:
            print(f"Video {name} has only {last_frame+1} frames !!")
            continue
        elif last_frame < SEQLEN-1:
            print(f"Video {name} has only {last_frame+1} frames.")
            data = np.concatenate([data, np.repeat(data[-1:], SEQLEN-last_frame-1, axis=0)], axis=0)
            last_frame = SEQLEN-1
        ## split data into train splits
        index_train = np.arange(0, last_frame, STRIDE)
        # remove the last indices if index[-i]+seqlen>last_frame
        while last_frame - index_train[-1] < SEQLEN-1:
            index_train = index_train[:-1]
        if last_frame - index_train[-1] >= MIN_REST-1:
            index_train = np.append(index_train, last_frame-SEQLEN)
        ## split the data into valid splits
        # index_valid = np.arange(0, last_frame, SEQLEN)
        # if last_frame - index_valid[-1] < SEQLEN-1:
        #     index_valid = index_valid[:-1]
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
            All_train.append(copy.deepcopy(d))
        # for tidx, c in enumerate(index_valid):
        #     start, end = c, c+SEQLEN
        #     cdata = copy.deepcopy(data[start:end])
        #     ## by default, take cast=='log_sref'
        #     for i in range(cdata.shape[0]):
        #         try:
        #             cdata[i] = CenteredScaled(cdata[i])
        #             if i==0:
        #                 ref_skel = copy.deepcopy(cdata[0])
        #             ## From the pre-shape space to a tangent space...
        #             cdata[i] = inv_exp(ref_skel, cdata[i])
        #         except:
        #             raise ValueError('!')
        #     d['data'] = cdata
        #     d['vid_name'] = name+f'*{tidx}'
        #     All_valid.append(copy.deepcopy(d))
        
    # dump the data
    fp_name = osp.basename(jsonpath).split('.')[0].split('_kinectv2')[0]
    save_fp_name = save_fp_format.format(fp_name, 'train') # 'test'
    # save_fp_valid = save_fp_format.format(fp_name, 'valid')
    with open(save_fp_name, 'wb') as f:
        pickle.dump(All_train, f)
    # with open(save_fp_valid, 'wb') as f:
    #     pickle.dump(All_valid, f)
