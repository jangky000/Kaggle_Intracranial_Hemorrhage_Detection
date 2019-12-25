#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2


# In[2]:


# import random

# sub = os.listdir('./png_images_128/stage_1_train_images')
# t_path=[]
# for x in range(len(sub)):
#     dir_path = './png_images_128/stage_1_train_images/'+sub[x]+'/'
#     t_path.append(dir_path)
    

# valid_list = random.sample(t_path, 3)
# train_list = list(random.sample(set(t_path) - set(valid_list), 7))


# In[3]:


import random
epidurals = pd.read_csv("epidurals.csv")["path"]
intraparenchymals = pd.read_csv("intraparenchymals.csv")["path"]
intraventriculars = pd.read_csv("intraventriculars.csv")["path"]
subarachnoids = pd.read_csv("subarachnoids.csv")["path"]
subdurals = pd.read_csv("subdurals.csv")["path"]
nothings = pd.read_csv("nothings.csv")["path"]


# In[4]:


train_list = []
valid_list = []

# sample_size = epidurals.shape[0]
# train_size = int(sample_size*0.7)
# valid_size = sample_size - train_size

sample_size = 1200
train_size = 1000
valid_size = 200

# print(sample_size*6)

# tmp = random.sample(list(epidurals), train_size)
# train_list.extend(tmp)
# valid_list.extend(list( set(epidurals) - set(tmp) ))

# def Sampling(diag):
#     tmp = random.sample(list(set(diag) - set(train_list)), train_size)
#     train_list.extend( tmp )
#     valid_list.extend(random.sample(list( set(diag) - set(tmp) - set(valid_list)), valid_size))

def Sampling(diag):
    tmp = list(set(diag) - set(train_list) - set(valid_list))[:sample_size]
    tmp2 = list(set(tmp))[:train_size]
    train_list.extend(tmp2)
    valid_list.extend(list(set(tmp) - set(tmp2)))


# def Sampling(diag):
#     tmp = random.sample(list(set(diag) - set(train_list) - set(valid_list)), sample_size)
#     tmp2 = random.sample(list(set(tmp)), train_size)
#     train_list.extend(tmp2)
#     valid_list.extend(list(set(tmp) - set(tmp2)))

Sampling(epidurals)
Sampling(intraparenchymals)
Sampling(intraventriculars)
Sampling(subarachnoids)
Sampling(subdurals)
Sampling(nothings)


print(len(train_list + valid_list))
print(len(set(train_list + valid_list)))
print(len(train_list))
print(len(set(train_list)))
print(len(valid_list))
print(len(set(valid_list)))
                
print(train_list[3])


# In[5]:


dir_path = './png_images_256/stage_1_train_images/'
class BrainDataset(Dataset):
    
    def __init__(self, data_name_list, csv_file):
        self.data_name_list = data_name_list
        self.df_name = pd.read_csv(csv_file)
        
        result_file_name=[]
        
        for i in range(len(self.data_name_list)):
            result_file_name.append(dir_path + self.data_name_list[i])
        
        self.result_file_name = result_file_name
        self.len = len(result_file_name)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        image = cv2.imread(self.result_file_name[idx])
        image = image[16:240, 16:240]
        
        
        image = image.astype(np.float)
        image = np.swapaxes(image, 0,2)
        
        result_str_list = self.result_file_name[idx].split("/")
        
        tmp1 = self.df_name[self.df_name['Image'] == result_str_list[4][:-4]]
        tmp2 = tmp1[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]].values
        tmp2 = tmp2.astype(np.float)
        tmp2 = tmp2.reshape(6)
                
        return torch.from_numpy(image).float(), torch.from_numpy(tmp2).float()
        


# In[6]:


dir_csv = './'
dir_train_img = './png_images_256/stage_1_train_images'
dir_test_img = './png_images_256/stage_1_test_images'
batch_size = 10


# In[7]:


train_dataset = BrainDataset(data_name_list=train_list, csv_file='train.csv')
valid_dataset = BrainDataset(data_name_list=valid_list, csv_file='train.csv')

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# In[8]:


batch = next(iter(data_loader_train))


# In[9]:



print("train len : ", len(data_loader_train))
print("valid len : ", len(data_loader_valid))


# In[10]:


batch[0].shape


# In[11]:


batch[1].shape


# In[12]:


print("train type : ", batch[0].type())
print("valid type : ", batch[0].type())


# In[13]:


#앞에 index는 image/label? 뒤에 index는 batch size 개수 중 index
plt.imshow(np.transpose(batch[0][0].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)


# In[14]:


import torch.nn as nn
import torch.nn.functional as F


# In[15]:


train_on_gpu = torch.cuda.is_available()
# train_on_gpu = 0
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


# In[16]:


device = torch.device('cuda:3')


# In[17]:


class Basic_CNN_3_VGG16_resdual_Net(nn.Module):
    def __init__(self):
        super(Basic_CNN_3_VGG16_resdual_Net, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.drop = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 6)
        

    def forward(self, x):
        
        x = F.relu(self.conv1_1(x))
        input1 = x
        x = self.pool(F.relu(self.conv1_2(x)) + input1)
        
        x = F.relu(self.conv2_1(x))
        input2 = x
        x = self.pool(F.relu(self.conv2_2(x)) + input2)
        
        x = F.relu(self.conv3_1(x))
        input3 = x
        x = F.relu(self.conv3_2(x))
        x = self.pool(F.relu(self.conv3_3(x)) + input3)
        
        x = F.relu(self.conv4_1(x))
        input4 = x
        x = F.relu(self.conv4_2(x))
        x = self.pool(F.relu(self.conv4_3(x)) + input4)
        
        x = F.relu(self.conv5_1(x))
        input5 = x
        x = F.relu(self.conv5_2(x))
        x = self.pool(F.relu(self.conv5_3(x)) + input5)
        
        x = x.view(-1, 25088)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.fc3(x)
        return x

model = Basic_CNN_3_VGG16_resdual_Net()
print(model)

if train_on_gpu:
    model.to(device)


# In[18]:


import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# In[19]:


'''만약 5개의 유형 중에 해당되는 병이 있다면, any는 무조건 true이다.'''
'''any가 0이라는 것은 5개의 유형의 병이 없다는 뜻이다.'''

classes = ["No Disease", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


# In[20]:


training_loss = []
validation_loss = []

train_avg_dif = []
valid_avg_dif = []

n_epochs = 120

valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    
    train_loss_difference = np.zeros(6)
    train_loss = 0.0
    valid_loss = 0.0
    
    t_cnt =1
    v_cnt =1
    model.train()
    '''training 갯수만큼 len(data_loader_train)만큼'''
    for data, target in data_loader_train:
        # 추가
#         data = autograd.Variable(data, volatile=True)
#         target = autograd.Variable(target, volatile=True)
        
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
#             train_loss_difference = train_loss_difference.cuda()
            
            
        optimizer.zero_grad()
        '''print(data)'''
        '''print(data.type())'''
        '''print(target.type())'''
        
        output = model(data)
        
        '''print(target.shape)'''
        
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
#         train_loss = train_loss.cpu()
        
#        print(train_loss)
        
        train_loss_difference = train_loss_difference + (output-target).abs().cpu().detach().numpy().sum(axis=0)/batch_size #메모리 누수
#         print("타입", type((output-target).abs().detach().numpy()))
#         print("train loss 차이", type(train_loss_difference), train_loss_difference)
        
#         if(t_cnt%1 == 0):
#             print('=' * 120)
#             print(t_cnt, " train output 결과값 : ", output)
#             print(t_cnt, " train ground trust 결과값 : ", target)
            
        t_cnt = t_cnt + 1
        
    model.eval()

    valid_loss_difference = np.zeros(6)
    '''valid 갯수만큼 len(data_loader_valid)만큼'''
    for data, target in data_loader_valid:
        
        
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
#             valid_loss_difference = valid_loss_difference.cuda()
        
        
        output = model(data)
        
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        
#         valid_loss = valid_loss.cpu()
#        print(valid_loss)
        
#         valid_loss += loss
        
        valid_loss_difference = valid_loss_difference + (output-target).abs().cpu().detach().numpy().sum(axis=0)/batch_size
#         print("valid loss 차이", type(train_loss_difference), train_loss_difference)
#         if(v_cnt%1 == 0):
#             print('=' * 120)
#             print(v_cnt," valid output 결과값 : ", output)
#             print(v_cnt," valid ground trust 결과값 : ", target)
            
        v_cnt = v_cnt + 1
    
    
    train_loss = train_loss/len(data_loader_train)
    valid_loss = valid_loss/len(data_loader_valid)
    
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    print("Average of differences", '='*30)
    print("Epoch : ", epoch, " train avg difference of 6 classes ", train_loss_difference/len(data_loader_train))
    print("Epoch : ", epoch, " valid avg difference of 6 classes ", valid_loss_difference/len(data_loader_valid))
    print('='*30)
    train_avg_dif.append(train_loss_difference/len(data_loader_train))
    valid_avg_dif.append(valid_loss_difference/len(data_loader_train))
    
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar4.pt')
        valid_loss_min = valid_loss


# In[ ]:


plt.plot(training_loss, label='Train loss')
plt.plot(validation_loss, label='Validation loss')
plt.legend(frameon=False)


# In[ ]:


train_avg_dif = np.array(train_avg_dif)
# print(train_avg_dif)
# print(train_avg_dif.T)

valid_avg_dif = np.array(valid_avg_dif)


# In[ ]:


for i in range(0, 6):
    plt.plot(train_avg_dif.T[i], label='Train Avg Dif' + str(i))

plt.legend(frameon=False)


# In[ ]:


for i in range(0, 6):
    plt.plot(valid_avg_dif.T[i], label='Valid Avg Dif'+str(i))
plt.legend(frameon=False)


# In[21]:


training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)
train_avg_dif = np.array(train_avg_dif)
valid_avg_dif = np.array(valid_avg_dif)


# In[22]:


np.save("./cnn3/training_loss.npy", training_loss)
np.save("./cnn3/validation_loss.npy", validation_loss)
np.save("./cnn3/train_avg_dif.npy", train_avg_dif)
np.save("./cnn3/valid_avg_dif.npy", valid_avg_dif)


# In[ ]:




