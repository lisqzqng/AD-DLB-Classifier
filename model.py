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
from collections import defaultdict

import time
import joblib
import warnings
warnings.filterwarnings("ignore")

import re

dic = {'updrs':4,'diag':5,'diag_3cls':3,'ad':2, 'updrs_3cls':3} 
os.system('clear')

######## Arguments ########
OUTPUT_DIR = os.path.join('results', time.strftime('%d-%m-%y_%Hh%Mm%S'))
os.makedirs(OUTPUT_DIR)
seed=4096
CKPT_PATH = 'models/checkpoints/AMAI_model_xsub_RigidTransform_NTU94_log_sref.pt' #Checkpoints path
POSTFIX = '-amai'
# split_name_fp = "./skeletons/split_names.json"
# split_name_fp = "./skeletons/miccai_split_names.json"
split_name_fp = "./skeletons/amai_split_names.json"
split_names = joblib.load(split_name_fp)
## LOSOCV
# subjects = [2,6,7,8,10,11,12,13,14,15]
# create the split names
# split_names = defaultdict(dict)
# for i in range(len(subjects)):
#     split_names[i]['val'] = ['Subject_'+str(subjects[i])+'_Camera{:d}'.format(j) for j in range(1,7)]
#     split_names[i]['train'] = []
#     for j in range(len(subjects)):
#         if j!=i:
#             split_names[i]['train'].extend(['Subject_'+str(subjects[j])+'_Camera{:d}'.format(k) for k in range(1,7)])
enable_ncv = False # Enable `n_cv` times nfold cross-validation, average over `n_cv` for final results
if enable_ncv:
    n_cv = len(split_names.keys())
else:
    split_names = [split_names]
    n_cv = 1
FOLD = len(split_names[0].keys()) # total number of folds per experiment
######## End of arguments ##########

torch.manual_seed(seed)
def load_model(num_classes, calculate_params=False):
    model = RigidNet120(mod,num_frames,num_joints).to('cpu')
    # original saved file with DataParallel
    checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    try:
        state_dict = checkpoint.state_dict() #['state_dict']
    except AttributeError:
        state_dict = checkpoint['state_dict']

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
num_frames=100
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
sklt_names = set([ sk.split('*')[0] for sk in os.listdir(f'./skeletons/{FOLD}-fold{POSTFIX}/') if sk.endswith('.pkl')])
for sklt_name in sklt_names:
    if 'pare' in sklt_name: continue
    for cls_name in ['ad','updrs_3cls', 'diag_3cls',]: # 'diag_3cls', 'ad', 'updrs_3cls'
        num_classes=dic[cls_name]

        # load the dataset from the pkl file
        with open(f'./skeletons/{FOLD}-fold{POSTFIX}/{sklt_name}*train.pkl', 'rb') as f:
            All_train_data = pickle.load(f)
        # map a fuction over the list of dictionary to add the vidname key
        All_train_data = list(map(lambda x: {**x, 'vidname': re.sub(r'_CC\d+', '', x['vid_name']).replace('f','').split('*')[0]}, All_train_data))

        try:
            with open(f'./skeletons/{FOLD}-fold{POSTFIX}/{sklt_name}*test.pkl', 'rb') as f:
                All_test_data = pickle.load(f)
        except FileNotFoundError:
            All_test_data = All_train_data
        else:
            All_test_data = list(map(lambda x: {**x, 'vidname': re.sub(r'_CC\d+', '', x['vid_name']).replace('f','').split('*')[0]}, All_test_data))

        names_train, names_test=[], []
        frames_train, frames_test=[], []
        data_train, data_test=[], []
        labels_train, labels_test=[],[]
        # conf_score=[]
        
        #choose interpolation or slicing
        # default: combine gait score 2 with 3
        label_cl=range(num_classes)
        # score=[0 for i in range(num_classes)]

        parts_acc = []
        result_dict = {}
        for ncv in range(n_cv):
            all_preds, all_labels = [], []
            list_eval=[]
            mat_global=np.zeros((num_classes,num_classes))
            for fold in range(FOLD):
                ## Load the train/test name split
                max_test_pred, max_test_label = None, None
                # ---> 'f' has been removed from all data names in preprocessing
                train_name_list = [x.replace('f','').split('*')[0] for x in split_names[ncv][fold]['train']]
                test_name_list = [x.replace('f','').split('*')[0] for x in split_names[ncv][fold]['val']]
                model = load_model(num_classes, calculate_params=False)
                model = RigidNet120(mod, num_frames, num_joints, num_labels=num_classes).to('cpu')
                # Define the loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
                # Split of data
                X_train = [x['data'] for x in All_train_data if x['vidname'] in train_name_list]
                X_test = [x['data'] for x in All_test_data if x['vidname'] in test_name_list]

                # Preprocess the class label
                if cls_name=='updrs_3cls':
                    y_train = [ get_3cls_updrs(x['gait_score']) for x in All_train_data if x['vidname'] in train_name_list ]
                    y_test = [ get_3cls_updrs(x['gait_score']) for x in All_test_data if x['vidname'] in test_name_list ]
                elif cls_name=='updrs':
                    y_train = [ x['gait_score'] for x in All_train_data if x['vidname'] in train_name_list ]
                    y_test = [ x['gait_score'] for x in All_test_data if x['vidname'] in test_name_list ]
                elif cls_name=='diag_3cls':
                    y_train = [ get_3cls_diag(x['diag']) for x in All_train_data if x['vidname'] in train_name_list ]
                    y_test = [ get_3cls_diag(x['diag']) for x in All_test_data if x['vidname'] in test_name_list ]
                elif cls_name=='diag':
                    y_train = [ x['diag'] for x in All_train_data if x['vidname'] in train_name_list ]
                    y_test = [ x['diag'] for x in All_test_data if x['vidname'] in test_name_list ]
                elif cls_name=='ad':
                    y_train = [ get_ad(x['diag']) for x in All_train_data if x['vidname'] in train_name_list ]
                    y_test = [ get_ad(x['diag']) for x in All_test_data if x['vidname'] in test_name_list ]
                else: 
                    raise NotImplementedError(f'Unknown classification type {cls_name} !!!')
                            
                                    
                print('evaluation n° {}:'.format(fold))

                def inter(elem):
                    n=elem.shape[0]
                    x = np.arange(n) # generate 1D x values
                    cs = CubicSpline(x, elem, axis=0) # perform cubic interpolation along axis 0
                    x_new = np.linspace(0, n, num_frames) # generate new 1D x values
                    data = cs(x_new) # compute interpolated data
                    return data
                
                batch_size = 32
                training_epochs=300
                steps = ceil(len(X_train) / batch_size)
                # writer = SummaryWriter()
                print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))
                # Train the model
                predict_tot=[]
                xx,yy=[],[]

                max_test_acc = -1.
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
                    batch_size_test=16
                    a=0
                    test_predict_tot=[]
                    xx,yy=[],[]
                    
                    with torch.no_grad():
                        correct_test=0
                        total_test=0
                        steps_test = ceil(len(X_test) / batch_size_test)
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
                            yy+=y
                    test_accuracy = 100*correct_test/total_test
                    print("Epoch {}: Test Accuracy = {} ".format(epoch+1,test_accuracy))
                    if test_accuracy>max_test_acc:
                        max_test_pred = test_predict_tot
                        max_test_label = y_test
                        max_test_acc = test_accuracy

                list_eval.append(max_test_acc)
                all_labels.append(max_test_label)
                all_preds.append(max_test_pred)
            # ================================ After n-fold training/validation ================================== #
            # print('List of accuracy:',list_eval)
            accuracy_mean = sum(list_eval)/FOLD
            print(f'Mean accuracy of the {sklt_name} on N_cv={ncv}', accuracy_mean,'%')
            out_string = '%s accuracy of %s: %.3f %%\n' % (cls_name, sklt_name.split('.')[0], accuracy_mean)
            # -----> calculate F1-score from confusion matrix
            mat_global = confusion_matrix(np.hstack(all_labels), np.hstack(all_preds))
            f1_score = np.zeros((len(label_cl),))
            for ci in range(len(label_cl)):
                f1_score[ci] = 2 * mat_global[ci, ci] / (np.sum(mat_global[ci, :]) + np.sum(mat_global[:, ci]))
            # calculate class average recal  and precision
            recall = np.nan_to_num(np.diag(mat_global) / np.sum(mat_global, axis=1))
            precision = np.nan_to_num(np.diag(mat_global) / np.sum(mat_global, axis=0))
            # calculate f1 score from precision and recall
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            # weighted F1-score
            w_f1_score = np.sum(f1_score * np.sum(mat_global, axis=1) / np.sum(mat_global))
            print(f'F1-score of the {sklt_name} for each class on N_cv={ncv}:', f1_score.mean())
            result_dict[ncv] = {'accuracy': accuracy_mean, 'min-max': max(list_eval)-min(list_eval),\
                'recall': recall.mean(), 'precision': precision.mean(), 'f1_score': f1_score.mean(), 'w_f1_score': w_f1_score}
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
        df.to_excel(os.path.join(OUTPUT_DIR, f"{cls_name}_{sklt_name.split('.')[0]}.xlsx"))
# print("F1_score of the network",score/10)    
print('Finished fine-tuning!')