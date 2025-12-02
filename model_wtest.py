import torch
import torch.nn as nn
import os

from models.RigidNet120 import RigidNet120
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import CubicSpline
from statistics import mean
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from math import ceil

import pickle
import pandas as pd
import json
from collections import defaultdict

import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import re

import os
import os.path as osp

dic = {'updrs':4,'diag':5,'diag_3cls':3,'ad':2, 'updrs_3cls':3} 
os.system('clear')
# OUTPUT_DIR = os.path.join('results', time.strftime('%d-%m-%y_%Hh%Mm%S'))
OUTPUT_DIR = './results/JBHI-rev3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
seed=4096
EPOCH=500
CKPT_PATH = './models/checkpoints/model_xsub_RigidTransform_NTU94_log_sref.pth' #Checkpoints path
# split_name_fp = "./skeletons/split_names.json"
split_name_fp = "./skeletons/robtulip_split_dict.json"
postfix = '-jbhi' # postfix for skeleton folder path
with open(split_name_fp,'r') as f:
    split_names = json.load(f)
## LOSOCV
# subjects = [2,6,7,8,10,11,12,13,14,15]
# create the split names
# split_names = defaultdict(dict)
# for i in range(len(subjects)):
#     split_names[i]['val'] = ['Subject_'+str(subjects[i])+'*amera{:d}'.format(j) for j in range(1,7)]
#     split_names[i]['train'] = []
#     for j in range(len(subjects)):
#         if j!=i:
#             split_names[i]['train'].extend(['Subject_'+str(subjects[j])+'*amera{:d}'.format(k) for k in range(1,7)])
split_names = [split_names]
# n_cv = len(split_names.keys())
n_cv = 1
FOLD = len(split_names[0].keys())
########End of arguments##########

torch.manual_seed(seed)
def load_model(num_classes, calculate_params=False):
    model = RigidNet120(mod,num_frames,num_joints).to('cpu')
    # original saved file with DataParallel
    checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint.state_dict() #['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k]=v
    
    # comment this line to train from scratch
    model.load_state_dict(new_state_dict)

        
    # Modify the last layer of the pre-trained model
    # Determine the type of the last layer
    last_layer = model.fc2
    if isinstance(last_layer, nn.Sequential):
        last_layer = last_layer[-1]  # Get the last layer in the sequence

    # Replace the last layer with a new layer
    in_features = last_layer.in_features
    model.fc2[-1] = nn.Linear(in_features, num_classes)

    # Print the modified model
    #print(model)

    # freeze the weights of the pre-trained layers (optional)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.conv2.parameters():
        param.requires_grad = True
    for param in model.LSTM.parameters():
        param.requires_grad = True
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True

    # Print the total trainable model parameters
    if calculate_params:
        # only count the requires_grad parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {total_params}')

    return model
#########Set your arguments##########
# Load the pre-trained model
# path='./models/checkpoints/model_xsub_RigidTransform_NTU94_log_sref.pt' #Checkpoints path
mod= 'RigidTransform'
num_frames=70
num_joints=25
use_cuda=1 # cpu 0, gpu 1
learning_rate=0.0001
protocol= 'sl'  #inter or sl or sl_mv
# for devices
use_cuda = 1

cuda = torch.cuda.is_available()
device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
if device == 'cuda':
    print('Using CUDA')
    torch.cuda.empty_cache()
else:
    print('NOT using CUDA')

## define functions for label post-processing
def get_3cls_diag(label):
    if label==0:
        return 0
    elif label==1 or label==3:
        return 1 # DLB
    else:
        return 2 # AD
get_ad = lambda l: 1 if l>0 else 0
get_3cls_updrs = lambda l: l if l<3 else 2
sklt_names = set([ sk.split('*')[0] for sk in os.listdir(f'./skeletons/{FOLD}-fold{postfix}/') if sk.endswith('train.pkl')])

# load skeleton data for hold-out test
# ho_test_fp = f'skeletons/{FOLD}-fold/wham_toaga*test.pkl'
ho_test_fp = ''
# # load the hold-out test data
try:
    with open(ho_test_fp, 'rb') as f:
        ho_test_data = pickle.load(f)
except FileNotFoundError:
    print(f'Hold-out test data not found at {ho_test_fp}. Please check the file path.')
    ho_test_data = []

for sklt_name in sklt_names:
    for cls_name in ['updrs_3cls',]: # 'diag_3cls', 'ad', 'updrs_3cls'
        num_classes=dic[cls_name]
        EXP_OUT_DIR = osp.join(OUTPUT_DIR, osp.basename(sklt_name).split('_')[0]+'-'+cls_name)
        os.makedirs(EXP_OUT_DIR, exist_ok=True)

        # load the dataset from the pkl file
        with open(f'./skeletons/{FOLD}-fold{postfix}/{sklt_name}*train.pkl', 'rb') as f:
            All_data = pickle.load(f)
        # map a fuction over the list of dictionary to add the vidname key
        # All_train_data = list(map(lambda x: {**x, 'vidname': re.sub(r'*C\d+', '', x['vid_name']).replace('f','').split('*')[0]}, All_train_data))

        # with open(f'./skeletons/{FOLD}-fold{postfix}/{sklt_name}*valid.pkl', 'rb') as f:
        #     All_valid_data = pickle.load(f)
        # All_valid_data = list(map(lambda x: {**x, 'vidname': re.sub(r'*C\d+', '', x['vid_name']).replace('f','').split('*')[0]}, All_valid_data))

        # with open(f'./skeletons/{FOLD}-fold{postfix}/{sklt_name}*testd.pkl', 'rb') as f:
        #     All_test_data = pickle.load(f)
        # All_test_data = list(map(lambda x: {**x, 'vidname': re.sub(r'*C\d+', '', x['vid_name']).replace('f','').split('*')[0]}, All_valid_data))

        names_train, names_valid=[], []
        frames_train, frames_valid=[], []
        data_train, data_valid=[], []
        labels_train, labels_valid=[],[]
        # conf_score=[]
        
        #choose interpolation or slicing
        # default: combine gait score 2 with 3
        label_cl=range(num_classes)
        # score=[0 for i in range(num_classes)]

        parts_acc = []
        result_dict = {}
        for ncv in range(n_cv):
            all_val_label, all_val_pred = [], []
            list_val_acc=[]
            list_test_acc=[]
            eval_out_dict = defaultdict(list)
            mat_global=np.zeros((num_classes,num_classes))
            for fold in range(FOLD):
                assert n_cv==1, 'Not implemented for n_cv > 1'
                ckpt_outfp = osp.join(EXP_OUT_DIR, 'fold'+str(fold)+'.pth')
                if osp.isfile(ckpt_outfp):
                    print(f'Load checkpoint from {ckpt_outfp}..')
                    IS_TRAIN = False
                else:
                    IS_TRAIN = True
                ## Load the train/valid name split
                # max_valid_pred, max_valid_label = None, None
                # ---> 'f' has been removed from all data names in preprocessing
                train_name_list = [x.replace('f','').split('*')[0] for x in split_names[ncv][str(fold)]['train']]
                valid_name_list = [x.replace('f','').split('*')[0] for x in split_names[ncv][str(fold)]['val']]
                test_name_list = [x.replace('f','').split('*')[0] for x in split_names[ncv][str(fold)]['test']]
                model_load_start = time.time()
                model = load_model(num_classes, calculate_params=False)
                model = RigidNet120(mod, num_frames, num_joints, num_labels=num_classes).to('cpu')
                model_load_time = time.time() - model_load_start
                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
                # Split of data
                X_train = [x['data'] for x in All_data if x['vid_name'].split('*')[0] in train_name_list]
                X_valid = [x['data'] for x in All_data if x['vid_name'].split('*')[0] in valid_name_list]
                X_test = [x['data'] for x in All_data if x['vid_name'].split('*')[0] in test_name_list]
                X_hotest = [x['data'] for x in ho_test_data]

                # Preprocess the class label
                if cls_name=='updrs_3cls':
                    y_train = [ get_3cls_updrs(x['gait_score']) for x in All_data if x['vid_name'].split('*')[0] in train_name_list ]
                    y_valid = [ get_3cls_updrs(x['gait_score']) for x in All_data if x['vid_name'].split('*')[0] in valid_name_list ]
                    y_test = [ get_3cls_updrs(x['gait_score']) for x in All_data if x['vid_name'].split('*')[0] in test_name_list ]
                    y_ho_test = [ get_3cls_updrs(x['gait_score']) for x in ho_test_data]
                elif cls_name=='updrs':
                    y_train = [ x['gait_score'] for x in All_data if x['vid_name'].split('*')[0] in train_name_list ]
                    y_valid = [ x['gait_score'] for x in All_data if x['vid_name'].split('*')[0] in valid_name_list ]
                    y_test = [ x['gait_score'] for x in All_data if x['vid_name'].split('*')[0] in test_name_list ]
                    y_ho_test = [ x['gait_score'] for x in ho_test_data]
                elif cls_name=='diag_3cls':
                    y_train = [ get_3cls_diag(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in train_name_list ]
                    y_valid = [ get_3cls_diag(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in valid_name_list ]
                    y_test = [ get_3cls_diag(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in test_name_list ]
                    y_ho_test = [ get_3cls_diag(x['diag']) for x in ho_test_data]
                elif cls_name=='diag':
                    y_train = [ x['diag'] for x in All_data if x['vid_name'].split('*')[0] in train_name_list ]
                    y_valid = [ x['diag'] for x in All_data if x['vid_name'].split('*')[0] in valid_name_list ]
                    y_test = [ x['diag'] for x in All_data if x['vid_name'].split('*')[0] in test_name_list ]
                    y_ho_test = [ x['diag'] for x in ho_test_data]
                else:
                    y_train = [ get_ad(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in train_name_list ]
                    y_valid = [ get_ad(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in valid_name_list ]
                    y_test = [ get_ad(x['diag']) for x in All_data if x['vid_name'].split('*')[0] in test_name_list ]
                    y_ho_test = [ get_ad(x['diag']) for x in ho_test_data]
                            
                                    
                print('evaluation n° {}:'.format(fold))

                def inter(elem):
                    n=elem.shape[0]
                    x = np.arange(n) # generate 1D x values
                    cs = CubicSpline(x, elem, axis=0) # perform cubic interpolation along axis 0
                    x_new = np.linspace(0, n, num_frames) # generate new 1D x values
                    data = cs(x_new) # compute interpolated data
                    return data
                
                batch_size = 32
                training_epochs=EPOCH
                steps = ceil(len(X_train) / batch_size)
                # writer = SummaryWriter()
                print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))
                # Train the model
                predict_tot=[]

                max_valid_acc = -1.
                if IS_TRAIN:
                    for epoch in range(training_epochs):
                        correct=0
                        total=0
                        running_loss = 0.0
                        epoch_loss = 0.0

                        for i in range(steps):
                            if (i*batch_size)+batch_size > len(X_train):
                                x, y = X_train[i*batch_size:], y_train[i*batch_size:]
                            else:
                                x, y = X_train[i*batch_size:(i*batch_size)+batch_size], y_train[i*batch_size:(i*batch_size)+batch_size]
                            trainx, trainy = np.array(x), np.array(y)
                            inputs, label = torch.from_numpy(trainx).to(device), torch.from_numpy(trainy).to(device)
                            model.to(device)
                            
                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Forward pass
                            outputs = model(inputs.float())
                            loss = criterion(outputs, label.long())

                            # Backward pass and optimize
                            loss.backward()
                            optimizer.step()

                            y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                            _, predicted = torch.max(y_pred_softmax, 1)
                            total += label.size(0)
                            correct += (predicted == label.long()).sum().item()
                            predict_tot+=predicted.cpu().tolist()
                            running_loss += loss.item()
                            epoch_loss += loss.item()
                            #print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss ))
                            running_loss = 0.0 

                        
                        accuracy = 100 * correct / total
                        epoch_loss = epoch_loss / steps
                        # writer.add_scalar('Loss/train', epoch_loss, epoch)
                        # writer.add_scalar('Accuracy/train', accuracy, epoch)
                        print("Epoch {}: Training Accuracy = {} : Training Loss {}".format(epoch+1,accuracy,epoch_loss))
                        # Evaluate the model on the validation set
                        batch_size_valid=16
                        a=0
                        valid_predict_tot=[]
                        
                        with torch.no_grad():
                            correct_valid=0
                            total_valid=0
                            steps_valid = ceil(len(X_valid) / batch_size_valid)
                            for i in range(steps_valid):
                                if (i*batch_size_valid)+batch_size_valid > len(X_valid):
                                    x, y = X_valid[i*batch_size_valid:], y_valid[i*batch_size_valid:]
                                else:                                  
                                    x, y = X_valid[i*batch_size_valid:(i*batch_size_valid)+batch_size_valid], y_valid[i*batch_size_valid:(i*batch_size_valid)+batch_size_valid]
                                validx, validy = np.array(x), np.array(y)
                                all_val_label.append(validy)
                                inputs, label = torch.from_numpy(validx).to(device), torch.from_numpy(validy).to(device)
                                outputs = model(inputs.float())
                                y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                                _, predicted = torch.max(y_pred_softmax, 1)
                                valid_predict_tot+=predicted.cpu().tolist()
                                total_valid += label.size(0)
                                correct_valid += (predicted == label.long()).sum().item()
                                all_val_pred.append(predicted.cpu().tolist())
                        valid_accuracy = 100*correct_valid/total_valid
                        print("Epoch {}: valid Accuracy = {} ".format(epoch+1,valid_accuracy))
                        if valid_accuracy>max_valid_acc:
                            # max_valid_pred = valid_predict_tot
                            # max_valid_label = y_valid
                            max_valid_acc = valid_accuracy
                            # save the model
                            torch.save(model, ckpt_outfp)
                    # load the model and evaluate 
                    model = torch.load(ckpt_outfp, map_location='cpu').to(device)
                    model.eval()   
                else:
                    # load the model and evaluate 
                    # # calculate the average inference time
                    model = torch.load(ckpt_outfp, map_location='cpu').to(device)
                    start_time = time.time()
                    model.eval()   
                    with torch.no_grad():
                        batch_size_valid=1
                        correct_valid=0
                        total_valid=0
                        steps_valid = ceil(len(X_valid) / batch_size_valid)
                        for i in range(steps_valid):
                            if (i*batch_size_valid)+batch_size_valid > len(X_valid):
                                x, y = X_valid[i*batch_size_valid:], y_valid[i*batch_size_valid:]
                            else:                                  
                                x, y = X_valid[i*batch_size_valid:(i*batch_size_valid)+batch_size_valid], y_valid[i*batch_size_valid:(i*batch_size_valid)+batch_size_valid]
                            validx, validy = np.array(x), np.array(y)
                            all_val_label.append(validy)
                            inputs, label = torch.from_numpy(validx).to(device), torch.from_numpy(validy).to(device)
                            outputs = model(inputs.float())
                            y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                            _, predicted = torch.max(y_pred_softmax, 1)
                            total_valid += label.size(0)
                            correct_valid += (predicted == label.long()).sum().item()
                            all_val_pred.append(predicted.cpu().tolist())
                    total_time = time.time() - start_time + model_load_time
                    total_frames = num_frames*len(X_valid)
                    print("Fold {}: evaluate time per frame: {}ms".format(fold, total_time/total_frames*1e3))
                    max_valid_acc = 100*correct_valid/total_valid
                    print("Fold {}: max valid Accuracy = {} ".format(fold,max_valid_acc))

                list_val_acc.append(max_valid_acc)
                # load the model and evaluate     
                batch_size_test = 16
                with torch.no_grad():
                    correct_test=0
                    total_test=0
                    steps_test = ceil(len(X_test) / batch_size_test)
                    test_predict_tot = []
                    test_gt_labels=[]
                    for i in range(steps_test):
                        if (i*batch_size_test)+batch_size_test > len(X_test):
                            x, y = X_test[i*batch_size_test:], y_test[i*batch_size_test:]
                        else:                                  
                            x, y = X_test[i*batch_size_test:(i*batch_size_test)+batch_size_test], y_test[i*batch_size_test:(i*batch_size_test)+batch_size_test]
                        testx, testy = np.array(x), np.array(y)
                        inputs, label = torch.from_numpy(testx).to(device), torch.from_numpy(testy).to(device)
                        outputs = model(inputs.float())
                        y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                        _, predicted = torch.max(y_pred_softmax, 1)
                        test_predict_tot+=predicted.cpu().tolist()
                        total_test += label.size(0)
                        correct_test += (predicted == label.long()).sum().item()
                        test_gt_labels+=y
                test_accuracy = 100*correct_test/total_test
                print("Fold {}: max test Accuracy = {} ".format(fold,test_accuracy))
                list_test_acc.append(test_accuracy)
                # calculate evaluation metrics at each fold
                cm = confusion_matrix(test_gt_labels, test_predict_tot, labels=list(label_cl))
                test_acc = np.diag(cm).sum() / cm.sum()
                test_precision = np.diag(cm) / (cm.sum(axis=0)+1e-8)
                test_recall = np.diag(cm) / (cm.sum(axis=1)+1e-8)
                test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall + 1e-8)
                # save evaluation metrics to the output dictionary
                eval_out_dict['fold'].append(fold)
                eval_out_dict['eval_name'].append('perfold')
                # eval_out_dict['val_accuracy'].append(max_valid_acc)
                eval_out_dict['test_accuracy'].append(test_accuracy)
                eval_out_dict['test_precision'].append(np.nanmean(test_precision[list(set(test_gt_labels))]))
                eval_out_dict['test_recall'].append(np.nanmean(test_recall[list(set(test_gt_labels))]))
                eval_out_dict['test_f1'].append(np.nanmean(test_f1[list(set(test_gt_labels))]))
                # -------> Evaluate on hold-out test set <------ #
                if len(X_hotest)>0:
                    with torch.no_grad():
                        correct_ho_test=0
                        total_ho_test=0
                        steps_ho_test = ceil(len(X_hotest) / batch_size_test)
                        ho_test_predict_tot = []
                        ho_test_gt_labels=[]
                        for i in range(steps_ho_test):
                            if (i*batch_size_test)+batch_size_test > len(X_hotest):
                                x, y = X_hotest[i*batch_size_test:], y_ho_test[i*batch_size_test:]
                            else:                                  
                                x, y = X_hotest[i*batch_size_test:(i*batch_size_test)+batch_size_test], y_ho_test[i*batch_size_test:(i*batch_size_test)+batch_size_test]
                            hotestx, hotesty = np.array(x), np.array(y)
                            inputs, label = torch.from_numpy(hotestx).to(device), torch.from_numpy(hotesty).to(device)
                            outputs = model(inputs.float())
                            y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                            _, predicted = torch.max(y_pred_softmax, 1)
                            ho_test_predict_tot+=predicted.cpu().tolist()
                            total_ho_test += label.size(0)
                            correct_ho_test += (predicted == label.long()).sum().item()
                            ho_test_gt_labels+=y
                    ho_test_accuracy = 100*correct_ho_test/total_ho_test
                    print("Hold-out test Accuracy = {} ".format(ho_test_accuracy))
                    cm_ho = confusion_matrix(ho_test_gt_labels, ho_test_predict_tot, labels=list(label_cl))
                    ho_test_precision = np.diag(cm_ho) / (cm_ho.sum(axis=0)+1e-8)
                    ho_test_recall = np.diag(cm_ho) / (cm_ho.sum(axis=1)+1e-8)
                    ho_test_f1 = 2 * ho_test_precision * ho_test_recall / (ho_test_precision + ho_test_recall + 1e-8)
                    eval_out_dict['fold'].append(fold)
                    eval_out_dict['eval_name'].append('holdout')
                    eval_out_dict['test_accuracy'].append(ho_test_accuracy)
                    eval_out_dict['test_precision'].append(np.nanmean(ho_test_precision[list(set(ho_test_gt_labels))]))
                    eval_out_dict['test_recall'].append(np.nanmean(ho_test_recall[list(set(ho_test_gt_labels))]))
                    eval_out_dict['test_f1'].append(np.nanmean(ho_test_f1[list(set(ho_test_gt_labels))]))

            # ================================ After 10-fold training/validation ================================== #
            # print('List of accuracy:',list_val_acc)
            val_accuracy_mean = sum(list_val_acc)/FOLD
            accuracy_mean = sum(list_test_acc)/FOLD
            print(f'Mean validation accuracy of the {sklt_name}', val_accuracy_mean,'%')
            out_string = '%s valiadation accuracy of %s: %.3f %%\n' % (cls_name, sklt_name.split('.')[0], val_accuracy_mean)
            print(f'Mean test accuracy of the {sklt_name}', accuracy_mean,'%')
            out_string += '%s test accuracy of %s: %.3f %%\n' % (cls_name, sklt_name.split('.')[0], accuracy_mean)
            # -----> calculate F1-score from confusion matrix
            mat_global = confusion_matrix(np.hstack(all_val_label), np.hstack(all_val_pred), labels=list(label_cl))
            val_f1_score = np.zeros((len(label_cl),))
            for ci in range(len(label_cl)):
                val_f1_score[ci] = 2 * mat_global[ci, ci] / (np.sum(mat_global[ci, :]) + np.sum(mat_global[:, ci]))
            # calculate class average recal  and precision
            val_recall = np.nan_to_num(np.diag(mat_global) / np.sum(mat_global, axis=1))
            val_precision = np.nan_to_num(np.diag(mat_global) / np.sum(mat_global, axis=0))
            # calculate f1 score from precision and recall
            val_f1_score = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-8)
            # weighted F1-score
            val_wf1_score = np.sum(val_f1_score * np.sum(mat_global, axis=1) / np.sum(mat_global))
            print(f'Validation F1-score of the {sklt_name} for each class on N_cv={ncv}:', val_f1_score.mean())
            result_dict[ncv] = {'val_accuracy': val_accuracy_mean, 'val_min-max': max(list_val_acc)-min(list_val_acc),\
                'val_recall': val_recall.mean(), 'val_precision': val_precision.mean(), 'val_f1_score': val_f1_score.mean(), 'val_wf1_score': val_wf1_score}
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = mat_global, display_labels = label_cl)
        # cm_display.plot()
        # cm_display.ax_.set_title(f'Confusion matrix of {sklt_name} for {cls_name}')
        # plt.show()
        # plt.savefig(os.path.join(OUTPUT_DIR, f"{sklt_name.split('.')[0]}_{cls_name}.png"))
        # with open(os.path.join(OUTPUT_DIR, f'{cls_name}.txt'), 'a') as fp:
        #     fp.write(out_string)
        #     fp.write(f"F1-score of the {sklt_name} for each class: {'  '.join(f1_score.astype(str).tolist())}\n")
        #     fp.write(f"Mean F1-score of the {sklt_name}: {f1_score.mean()}\n")
        
        # # write result dictionary to excel
        df = pd.DataFrame(result_dict)
        df = df.T
        df.to_excel(os.path.join(OUTPUT_DIR, f"val_{cls_name}_{sklt_name.split('.')[0]}.xlsx"))

        # save evaluation metrics to xlsx
        eval_out_df = pd.DataFrame(eval_out_dict)
        eval_out_df.to_excel(os.path.join(EXP_OUT_DIR, f"{cls_name}_{sklt_name.split('.')[0]}_eval.xlsx"), index=False)
# print("F1_score of the network",score/10)    
print('Finished fine-tuning!')