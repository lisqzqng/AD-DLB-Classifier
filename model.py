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

#########Set your arguments##########

# Load the pre-trained model
path='models/checkpoints/model_xsub_RigidTransform_NTU94_log_sref.pt' #Checkpoints path
mod= 'RigidTransform'
num_frames=100
num_joints=25
use_cuda=0 # cpu 0, gpu 1
learning_rate=0.0001
protocol= 'sl'  #inter or sl or sl_mv
num_classes=3
mv=0 #if 1 with majority voting, 0 if not
gs=1 #0 if gait score 0,1,2,3 ,1 if gait score 0,1,2/3
num_over=50 #
# Load the dataset from the pkl file
with open('Baseline.pkl', 'rb') as f:
    All_data = pickle.load(f)
seed=4096
########End of arguments##########

torch.manual_seed(seed)
model = RigidNet120(mod,num_frames,num_joints).to('cpu')
# original saved file with DataParallel
checkpoint = torch.load(path, map_location=torch.device('cpu'))
state_dict =checkpoint['state_dict']

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


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    
names=[]
frames=[]
data=[]
labels=[]
conf_score=[]

for i in All_data:
    names.append(i['vid_name'])
    frames.append(i['num_frames'])
    data.append(i['data'])
    #conf_score.append(i['conf_score'])
    if num_classes==4 and gs==1:
        if i['gait_score']>=2:
            labels.append(2)
        else:
            labels.append(i['gait_score'])
    
    elif num_classes==4 and gs==0:
        labels.append(i['gait_score'])
          
    elif num_classes==3:
        if i['diag']==0:
            labels.append(0)
        elif i['diag']==2 or i['diag']==4:
            labels.append(1)
        else:
            labels.append(2)
            
    elif num_classes==2:
        if i['diag']==0:
            labels.append(0)
        else:
            labels.append(1)
            
    else: #==5
        labels.append(i['diag'])
    
use_cuda = 1

cuda = torch.cuda.is_available()
device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
if device == 'cuda':
    print('Using CUDA')
    torch.cuda.empty_cache()
else:
    print('NOT using CUDA')


#choose interpolation or slicing
parts=10
list_eval=[]
list_net=[]
name_pred=[]
if num_classes!=4 or (num_classes==4 and gs==0):
    label_cl=range(num_classes)
    mat_global=np.zeros((num_classes,num_classes))
    score=[0 for i in range(num_classes)]
elif num_classes==4 and gs==1:
    label_cl=range(num_classes-1)
    mat_global=np.zeros((num_classes-1,num_classes-1))
    score=[0 for i in range(num_classes-1)]


for i in range(parts):
    per=int(len(data)/parts)
    # Split of data
    if i!=9:
        X_test, y_test=data[i*per:(i*per)+per], labels[i*per:(i*per)+per]
        N_test=names[i*per:(i*per)+per]
        #conf_test=conf_score[i*per:(i*per)+per]
        X_train, y_train= data.copy(), labels.copy()
        del X_train[i*per:(i*per)+per]
        del y_train[i*per:(i*per)+per]
    else:
        X_test, y_test=data[i*per:], labels[i*per:]
        N_test=names[i*per:]
        X_train, y_train= data.copy(), labels.copy()
        del X_train[i*per:]
        del y_train[i*per:]
    print('evaluation n° {}:'.format(i+1))

    def inter(elem):
        n=elem.shape[0]
        x = np.arange(n) # generate 1D x values
        cs = CubicSpline(x, elem, axis=0) # perform cubic interpolation along axis 0
        x_new = np.linspace(0, n, num_frames) # generate new 1D x values
        data = cs(x_new) # compute interpolated data
        return data
    
    list=[[X_train,y_train],[X_test,y_test]]
    list_new1=[]

    #interpolation
    for i in list:
        indx=[]
        new=[]
        for j in i[0]:
            data_inter=inter(j)
            indx.append(data_inter)
        new=[indx,i[1]]
        list_new1.append(new)
            
        
    #sliding
    list_new2=[]
    for i in list:
        data_new=[]
        
        label_new2=[]
        num=[] #list of number de window
        for c , arr in enumerate(i[0]):
            #label of sample
            l=i[1][c]
            
            # calculate the number of windows that can be extracted
            n_windows = (arr.shape[0] - num_frames) // num_over + 1
            if n_windows < 0:
                n_windows=0
                
            # loop over the array and extract the windows
            for j in range(n_windows):
                window = arr[j*num_over:j*num_over+num_frames,:,:]
                data_new.append(window)
            
            #if the num of frames < 100
            if n_windows!=0:
                num.append(n_windows)
            """elif ((arr.shape[0]-(j*num_over+num_frames)) >= 2):
                window=inter(arr[j*num_over+num_frames:,:,:])
                data_new.append(window)
                n_windows+=1"""
            
            #duplicate num labels
            for k in range(n_windows):
                label_new2.append(l)
                
            
        t=[data_new,label_new2]
        list_new2.append(t)
        
    if protocol=='inter':    
        X_train, y_train, X_test, y_test= list_new1[0][0], list_new1[0][1],list_new1[1][0],list_new1[1][1]
    elif protocol in ['sl','sl_mv']:
        X_train, y_train, X_test, y_test= list_new2[0][0], list_new2[0][1],list_new2[1][0],list_new2[1][1] 
    elif protocol=='both':
        X_train, y_train, X_test, y_test= list_new2[0][0]+list_new1[0][0], list_new2[0][1]+list_new1[0][1],list_new2[1][0]+list_new1[1][0],list_new2[1][1]+list_new1[1][1]
    else:
        X_train, y_train, X_test, y_test= list_new2[0][0]+list_new1[0][0], list_new2[0][1]+list_new1[0][1],list_new1[1][0],list_new1[1][1] #only inter for test
    
    
    batch_size = 256
    training_epochs=100
    steps = int(len(X_train) / batch_size)
    writer = SummaryWriter()
    print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))
    # Train the model
    predict_tot=[]
    xx,yy=[],[]
    
    for epoch in range(training_epochs):
        correct=0
        total=0
        running_loss = 0.0
        epoch_loss = 0.0
        
        for i in range(steps):
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
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print("Epoch {}: Training Accuracy = {} : Training Loss {}".format(epoch+1,accuracy,epoch_loss))
    # Evaluate the model on the validation set
    batch_size_test=16
    a=0
    predict_tot=[]
    xx,yy=[],[]
    
    if mv==0:
        with torch.no_grad():
            correct_test=0
            total_test=0
            steps = int(len(X_test) / batch_size_test)
            for i in range(steps):                        
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
        accuracy = 100*correct_test/total_test
        list_eval.append(accuracy)
        print('Accuracy of the network : %.3f %%' % accuracy)
        mat1 =confusion_matrix(yy,predict_tot,labels=label_cl)
        mat_global+=mat1
        sc=f1_score(yy, predict_tot, labels=label_cl, average='macro')
        print("F1_score",sc)
        score+=sc
        #print("tn:{}, fp:{}, fn:{}, tp:{}".format(tn, fp, fn, tp))
        print(mat1)
    else:
        with torch.no_grad():
            correct_test=0
            total_test=0
            test=0
            predict_tot=[]
            prob=[]
            for j in num:                        
                x, y = X_test[a:a+j], y_test[a:a+j]
                a+=j
                total_test += label.size(0)
                testx, testy = np.array(x), np.array(y)
                inputs, label = torch.from_numpy(testx).to(device), torch.from_numpy(testy).to(device)                
                outputs = model(inputs.float())
                y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                _, predicted = torch.max(y_pred_softmax, 1)
                predict_tot+=predicted.cpu().tolist()
                correct_test += (predicted == label.long()).sum().item()
                pred=np.array(predicted.cpu()).tolist()
                cl0=pred.count(0)
                cl1=pred.count(1)
                cl2=pred.count(2)
                cl3=pred.count(3)
                dict={cl0:0,cl1:1,cl2:2,cl3:3}
                maxi=max(dict.keys())
                cl=dict[maxi]  #majority voting
                if label[0]==cl:
                    test+=1
     
                xx.append(cl)
                yy.append(label.cpu().tolist()[0])
            
        mat=confusion_matrix(yy,xx,labels=label_cl)
        #mat1 =confusion_matrix(y_test,predict_tot,labels=label_cl)
        mat_global+=mat
        #print("tn:{}, fp:{}, fn:{}, tp:{}".format(tn, fp, fn, tp))
        print(mat)
        #print('la matrice de probabilité:{}'.format(prob))
        accuracy = 100* test/ len(num)
        list_eval.append(accuracy)
        print('Accuracy of the majority voting:',accuracy,'%')
        net_acc=100 * correct_test / total_test
        list_net.append(net_acc)
        print('Accuracy of the network: %.3f %%' % net_acc)
if mv==1:
    print('List of accuracy of network:',list_net)
    print('List of accuracy of majority voting:',list_eval)
    net_accuracy=sum(list_net)/parts
    accuracy_mean=sum(list_eval)/parts
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = mat_global, display_labels = label_cl)
    cm_display.plot()
    plt.show() 
    print('Mean accuracy of the majority voting',accuracy_mean,'%')
    #print('Mean accuracy of the network',net_accuracy,'%')

else:
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = mat_global, display_labels = label_cl)
    cm_display.plot()
    plt.show()
    print('List of accuracy:',list_eval)
    accuracy_mean=sum(list_eval)/parts
    print('Mean accuracy of the network',accuracy_mean,'%')

print("F1_score of the network",score/10)    
print('Finished fine-tuning!')