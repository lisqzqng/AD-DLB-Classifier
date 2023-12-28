import torch
import torch.nn as nn
import os
import pickle
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

import time
import joblib
import warnings
warnings.filterwarnings("ignore")

dic = {'updrs':4,'diag':5,'ad':2,'updrs_3cls':3,'diag_3cls':3}
os.system('clear')
OUTPUT_DIR = os.path.join('results', time.strftime('%d-%m-%y_%Hh%Mm%S'))
os.makedirs(OUTPUT_DIR)
seed=4096
CKPT_PATH = './models/checkpoints/model_xsub_RigidTransform_NTU94_log_sref.pt' #Checkpoints path
split_name_fp = "/home/dw/Documents/DAMoS/Code/Vita-CLIP/datasets/hospital/split_names.json"
split_names = joblib.load(split_name_fp)
########End of arguments##########

torch.manual_seed(seed)
def load_model(num_classes):
    model = RigidNet120(mod,num_frames,num_joints).to('cpu')
    # original saved file with DataParallel
    checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k]=v
        
    model.load_state_dict(new_state_dict)
    #print(model)

        
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
    return model
#########Set your arguments##########
# Load the pre-trained model
# path='./models/checkpoints/model_xsub_RigidTransform_NTU94_log_sref.pt' #Checkpoints path
mod= 'RigidTransform'
num_frames=50
num_joints=25
use_cuda=1 # cpu 0, gpu 1
learning_rate=0.0001
protocol= 'sl'  #inter or sl or sl_mv
FOLD = 5
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
sklt_names = set([ sk.split('*')[0] for sk in os.listdir(f'./skeletons/{FOLD}-fold/')])
for sklt_name in sklt_names:
    for cls_name in ['updrs', 'updrs_3cls','diag', 'diag_3cls',]: # 'ad'
        num_classes=dic[cls_name]

        # load the dataset from the pkl file
        with open(f'./skeletons/{FOLD}-fold/{sklt_name}*train.pkl', 'rb') as f:
            All_train_data = pickle.load(f)

        with open(f'./skeletons/{FOLD}-fold/{sklt_name}*test.pkl', 'rb') as f:
            All_test_data = pickle.load(f)

        names_train, names_test=[], []
        frames_train, frames_test=[], []
        data_train, data_test=[], []
        labels_train, labels_test=[],[]
        # conf_score=[]
        
        #choose interpolation or slicing
        list_eval=[]
        list_net=[]
        name_pred=[]
        # default: combine gait score 2 with 3
        label_cl=range(num_classes)
        mat_global=np.zeros((num_classes,num_classes))
        score=[0 for i in range(num_classes)]

        parts_acc = []
        for fold in range(FOLD):
            ## Load the train/test name splitsi
            train_name_list = [x.split('.')[0] for x in split_names[fold+1]['train']]
            test_name_list = [x.split('.')[0] for x in split_names[fold+1]['test']]
            model = load_model(num_classes)
            # model = RigidNet120(mod, num_frames, num_joints).to('cpu')
            last_layer = model.fc2
            in_features = last_layer[-1].in_features
            model.fc2[-1] = nn.Linear(in_features, num_classes)
            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
            # Split of data
            X_train = [x['data'] for x in All_train_data if x['vid_name'] in train_name_list]
            X_test = [x['data'] for x in All_test_data if x['vid_name'] in test_name_list]
            # Preprocess the class label
            if cls_name=='updrs_3cls':
                y_train = [ get_3cls_updrs(x['gait_score']) for x in All_train_data if x['vid_name'] in train_name_list ]
                y_test = [ get_3cls_updrs(x['gait_score']) for x in All_test_data if x['vid_name'] in test_name_list ]
            elif cls_name=='updrs':
                y_train = [ x['gait_score'] for x in All_train_data if x['vid_name'] in train_name_list ]
                y_test = [ x['gait_score'] for x in All_test_data if x['vid_name'] in test_name_list ]
            elif cls_name=='diag_3cls':
                y_train = [ get_3cls_diag(x['diag']) for x in All_train_data if x['vid_name'] in train_name_list ]
                y_test = [ get_3cls_diag(x['diag']) for x in All_test_data if x['vid_name'] in test_name_list ]
            elif cls_name=='diag':
                y_train = [ x['diag'] for x in All_train_data if x['vid_name'] in train_name_list ]
                y_test = [ x['diag'] for x in All_test_data if x['vid_name'] in test_name_list ]
            else:
                y_train = [ get_ad(x['diag']) for x in All_train_data if x['vid_name'] in train_name_list ]
                y_test = [ get_ad(x['diag']) for x in All_test_data if x['vid_name'] in test_name_list ]
            
            print('evaluation n° {}:'.format(fold+1))

            def inter(elem):
                n=elem.shape[0]
                x = np.arange(n) # generate 1D x values
                cs = CubicSpline(x, elem, axis=0) # perform cubic interpolation along axis 0
                x_new = np.linspace(0, n, num_frames) # generate new 1D x values
                data = cs(x_new) # compute interpolated data
                return data
            
            batch_size = 16
            training_epochs=200
            steps = ceil(len(X_train) / batch_size)
            # writer = SummaryWriter()
            print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))
            # Train the model
            predict_tot=[]
            xx,yy=[],[]

            max_test_acc = 0.0
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
                batch_size_test=1
                a=0
                predict_tot=[]
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
                        predict_tot+=predicted.cpu().tolist()
                        total_test += label.size(0)
                        correct_test += (predicted == label.long()).sum().item()
                        yy+=y
                test_accuracy = 100*correct_test/total_test
                print("Epoch {}: Test Accuracy = {} ".format(epoch+1,test_accuracy))
                if test_accuracy>max_test_acc:
                    max_test_acc = test_accuracy
                    mat = confusion_matrix(yy,predict_tot,labels=label_cl)

            list_eval.append(max_test_acc)
            mat_global+=mat
        # ================================ After 10-fold training/validation ================================== #
        print('List of accuracy:',list_eval)
        accuracy_mean=sum(list_eval)/FOLD
        print(f'Mean accuracy of the {sklt_name}',accuracy_mean,'%')
        out_string = '%s accuracy of %s: %.3f %%\n' % (cls_name, sklt_name.split('.')[0], accuracy_mean)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = mat_global, display_labels = label_cl)
        cm_display.plot()
        cm_display.ax_.set_title(f'Confusion matrix of {sklt_name} for {cls_name}')
        # plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{sklt_name.split('.')[0]}_{cls_name}.png"))
        with open(os.path.join(OUTPUT_DIR, f'{cls_name}.txt'), 'a') as fp:
            fp.write(out_string)
# print("F1_score of the network",score/10)    
print('Finished fine-tuning!')