import joblib
import os 
import numpy as np
from numpy import ndarray
import pickle
import random

os.chdir('E:\KShapeNet-main')  #Set the path of your data .json

#info=joblib.load('vibe_skeleton-label_25-04-2023.json')
#info=joblib.load('VPare+_25-04-2023.json')

info = joblib.load("NoJWFF_tasc_skeleton-label_17-05-2023_raw.json")
#info=joblib.load('VPare+wSmooth_skeleton-label_28-04-2023.json')
#info=joblib.load('pare_wSmooth_skeleton-label_28-04-2023.json')
#info=joblib.load('vibe_wSmooth_skeleton-label_28-04-2023.json')
#info=joblib.load('pare_skeleton-label_25-04-2023.json')

names_total=list(info['vid_name'])
diag_total=list(info['Diag'])
data= list(info['joints3D'])
vid = list(dict.fromkeys(info['vid_name'])) 
gait=list(info['gait_score'])
#conf_score= list(info['confidence'])

All=[]
d={}
cl=2
a=0
label_new, data_new=[],[]


# Convert the list to a set to eliminate repetitions
unique_set = set(names_total)

# Convert the set back to a list
names = list(unique_set)

for name in names:  
    if names_total.count(name)>=100:
        d['vid_name']=name
        count=names_total.count(name)
        d['num_frames']=count
        d['diag']=diag_total[a]
        d['gait_score']=gait[a]
        d['data']=np.array(data[a:a+count])
        #d['conf_score']=conf_score[a:a+count]
        a+=count
        All.append(d)
        d={}
       

# load the data
with open('NoJWFF.pkl', 'wb') as f:
    pickle.dump(All, f)
