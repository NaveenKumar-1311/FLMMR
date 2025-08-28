import torch
from torchvision import utils as vutils, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from imutils import paths
from random import randint
import os
from PIL import Image as Image
from torchvision import models


import ot
import ot.plot

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():  
    cuda3 = torch.device('cuda:2') 
else:  
    cuda3 = torch.device('cpu') 


train_path = './Train_CIFAR10'
test_csv_file_path = './Test.csv'



# In[5]:


NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 32


# In[6]:


data_transforms = transformations = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[7]:


train_dataset = datasets.ImageFolder(train_path, transform=data_transforms)
num_train = len(train_dataset)
print('number of training samples :', num_train)


# In[8]:


train_sampler, test_sampler = torch.utils.data.random_split(train_dataset, (round(0.8*len(train_dataset)), round(0.2*len(train_dataset))))


# test_dataset = GTSRB(root_dir='.', csv_file=test_csv_file_path, transform= data_transforms)
print('number of training samples :', len(train_sampler))
print('number of testing samples :', len(test_sampler))



def create_clients(data, num_clients, initial, alpha):
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    training_data_size = len(data)

    # create a dirichlet distribution with concentration parameter alpha
    proportions = np.random.dirichlet(np.ones(num_clients)*alpha)

    # calculate the number of data samples for each client
    shard_sizes = (proportions * training_data_size).astype(int)
   
    shards = []
    start_idx = 0
    # assign data shards to each client
    for i in range(num_clients):
        shard = torch.utils.data.Subset(data, range(start_idx, start_idx + shard_sizes[i]))
        shards.append(shard)
        start_idx += shard_sizes[i]
        
    for i in range(len(shards)):
        print('client {}: data size: {}'.format(client_names[i], len(shards[i])))
        
    return {client_names[i]: shards[i] for i in range(len(client_names))}

clients = create_clients(train_sampler, num_clients=100,initial='clients', alpha=1)
print('Clients created..done')


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tf pytorch dataloaderds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        pytorch dataloader object'''

    
    trainloader = torch.utils.data.DataLoader(data_shard, batch_size=BATCH_SIZE,
                                            shuffle=False, drop_last= True, num_workers=2)
    
    return trainloader


# In[15]:


#process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    
#process and batch the test set  
test_loader = torch.utils.data.DataLoader(test_sampler, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2, drop_last=True)


# class CustomCNN(nn.Module):

#     def __init__(self):
#         super(CustomCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 5)
#         self.conv1_bn = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 128, 3)
#         self.conv2_bn = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, 1)
#         self.conv3_bn = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 256, 1)
#         self.conv4_bn = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(256 * 8 * 8, 512)
#         self.fc1_bn = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 43)
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.dropout(self.conv1_bn(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout(self.conv2_bn(x))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.conv3_bn(x)
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.dropout(self.conv4_bn(x))
#         x = x.view(-1, 256 * 8 * 8)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(self.fc1_bn(x))
#         x = F.softmax(self.fc2(x), -1)
#         return x

CustomCNN = models.resnet18()

#print(CustomCNN)

# print(CustomCNN.fc.out_features) # 1000 



# Freeze training for all layers
# for param in CustomCNN.features.parameters():
#     param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = CustomCNN.fc.in_features
#features = list(CustomCNN.fc.children())[:-1] # Remove last layer
#features.extend([nn.Linear(num_features, NUM_CLASSES)]) # Add our layer with 4 outputs
#CustomCNN.fc = nn.Sequential(*features) # Replace the model fc
CustomCNN.fc = nn.Linear(num_features, NUM_CLASSES)

# In[18]:


def scale_model_weights(client_models, weight_multiplier_list):
    '''function for scaling a models weights'''
    client_model = client_models[0].state_dict()
    for i in range(len(client_models)):
      for k in client_model.keys():
        client_models[i].state_dict()[k].float()*weight_multiplier_list[i]

    return client_models


# In[19]:


def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
#     for model in client_models:
#         model.load_state_dict(global_model.state_dict())
    return global_model


def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    global_model.eval()
    loss = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    test_size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cuda3), target.to(cuda3)
            output = global_model(data)
            test_loss += loss(output, target) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            del data
            del target

    test_loss /= test_size
    acc = correct / test_size

    return acc, test_loss
    
def maps_t(alpha_st, X):
    #warnings.filterwarnings('ignore')
    pt_s = alpha_st.T/alpha_st.sum(axis = 1)
    #print(pt_s, np.where(pt_s == 1)[1])
    pt_s[~ np.isfinite(pt_s)] = 0
    #print(X.T)
    Xs_mapped = np.matmul(X.T, pt_s)
    #print(Xs_mapped)
    return Xs_mapped.T

def otattackfrombeginning(aggregiatedmodel, g_epoch):
    val1a1=[]
    val1a2=[]
    val1a3=[]
    val1a4=[]
    val1a5=[]
    val1a6=[]
    val1a7=[]
    val1a8=[]
    val1a9=[]
    val1a10=[]
    val1a11=[]
    val1a12=[]
    val1a13=[]
    val1a14=[]
    val1a15=[]
    val1a16=[]
    val1a17=[]
    val1a18=[]


    for name, para in aggregiatedmodel.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('conv1.weight'):
            val1a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.0.conv1.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.0.conv2.weight'):
            val1a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.conv1.weight'):
            val1a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.conv2.weight'):
            val1a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.conv1.weight'):
            val1a6 = torch.flatten(para).cpu().detach().numpy()

        if name == str('layer2.0.conv2.weight'):
            val1a7 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.conv1.weight'):
            val1a8 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.conv2.weight'):
            val1a9 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.0.conv1.weight'):
            val1a10 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.0.conv2.weight'):
            val1a11 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.1.conv1.weight'):
            val1a12 = torch.flatten(para).cpu().detach().numpy()

        if name == str('layer3.1.conv2.weight'):
            val1a13 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.0.conv1.weight'):
            val1a14 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.0.conv2.weight'):
            val1a15 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.1.conv1.weight'):
            val1a16 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.1.conv2.weight'):
            val1a17 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val1a18 = torch.flatten(para).cpu().detach().numpy()

    model_path = "./globalmodel_base_30C1A_heter_4102023.pt"
    global_model3 = CustomCNN.to(cuda3)
    global_model3.load_state_dict(torch.load(model_path, map_location='cpu'))
    global_model3.eval()
 
    val3a1=[]
    val3a2=[]
    val3a3=[]
    val3a4=[]
    val3a5=[]
    val3a6=[]
    val3a7=[]
    val3a8=[]
    val3a9=[]
    val3a10=[]
    val3a11=[]
    val3a12=[]
    val3a13=[]
    val3a14=[]
    val3a15=[]
    val3a16=[]
    val3a17=[]
    val3a18=[]


    for name, para in global_model3.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('conv1.weight'):
            val3a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.0.conv1.weight'):
            val3a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.0.conv2.weight'):
            val3a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.conv1.weight'):
            val3a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.conv2.weight'):
            val3a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.conv1.weight'):
            val3a6 = torch.flatten(para).cpu().detach().numpy()

        if name == str('layer2.0.conv2.weight'):
            val3a7 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.conv1.weight'):
            val3a8 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.conv2.weight'):
            val3a9 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.0.conv1.weight'):
            val3a10 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.0.conv2.weight'):
            val3a11 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer3.1.conv1.weight'):
            val3a12 = torch.flatten(para).cpu().detach().numpy()

        if name == str('layer3.1.conv2.weight'):
            val3a13 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.0.conv1.weight'):
            val3a14 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.0.conv2.weight'):
            val3a15 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.1.conv1.weight'):
            val3a16 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer4.1.conv2.weight'):
            val3a17 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val3a18 = torch.flatten(para).cpu().detach().numpy()
    #****************************************************************************

    if g_epoch >= 0 and g_epoch <=20:
        # print("Entering CLP --> 1")
        n1 = 9408 #3969  # nb samples
        n2 = 5000
        n3 = 5000
        n4 = 5000
        n5 = 5000
        n6 = 2500

        n7 = 2500 #3969  # nb samples
        n8 = 2500
        n9 = 2500
        n10 = 1200
        n11 = 1200
        n12 = 1200

        n13 = 1200 #3969  # nb samples
        n14 = 1000
        n15 = 1000
        n16 = 1000
        n17 = 1000
        n18 = 5120

        # print(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18)
    elif g_epoch > 20 and g_epoch <=40:
        # print("Entering CLP --> 2")
        n1 = int(9408/2) #3969  # nb samples
        n2 = int(5000/2)
        n3 = int(5000/2)
        n4 = int(5000/2)
        n5 = int(5000/2)
        n6 = int(2500/2)

        n7 = int(2500/2) #3969  # nb samples
        n8 = int(2500/2)
        n9 = int(2500/2)
        n10 = int(1200/2)
        n11 = int(1200/2)
        n12 = int(1200/2)

        n13 = int(1200/2) #3969  # nb samples
        n14 = int(1000/2)
        n15 = int(1000/2)
        n16 = int(1000/2)
        n17 = int(1000/2)
        n18 = int(5120/2)
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 40 and g_epoch <=60:
        # print("Entering CLP --> 3")
        n1 = int(9408/4) #3969  # nb samples
        n2 = int(5000/4)
        n3 = int(5000/4)
        n4 = int(5000/4)
        n5 = int(5000/4)
        n6 = int(2500/4)

        n7 = int(2500/4) #3969  # nb samples
        n8 = int(2500/4)
        n9 = int(2500/4)
        n10 = int(1200/4)
        n11 = int(1200/4)
        n12 = int(1200/4)

        n13 = int(1200/4) #3969  # nb samples
        n14 = int(1000/4)
        n15 = int(1000/4)
        n16 = int(1000/4)
        n17 = int(1000/4)
        n18 = int(5120/4)
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 60 and g_epoch <=80:
        # print("Entering CLP --> 4")
        n1 = int(9408/8) #3969  # nb samples
        n2 = int(5000/8)
        n3 = int(5000/8)
        n4 = int(5000/8)
        n5 = int(5000/8)
        n6 = int(2500/8)

        n7 = int(2500/8) #3969  # nb samples
        n8 = int(2500/8)
        n9 = int(2500/8)
        n10 = int(1200/8)
        n11 = int(1200/8)
        n12 = int(1200/8)

        n13 = int(1200/8) #3969  # nb samples
        n14 = int(1000/8)
        n15 = int(1000/8)
        n16 = int(1000/8)
        n17 = int(1000/8)
        n18 = int(5120/8)
        # print(n1, n2, n3, n4, n5, n6)
    else:
        # print("Entering CLP --> 5")
        n1 = int(9408/16) #3969  # nb samples
        n2 = int(5000/16)
        n3 = int(5000/16)
        n4 = int(5000/16)
        n5 = int(5000/16)
        n6 = int(2500/16)

        n7 = int(2500/16) #3969  # nb samples
        n8 = int(2500/16)
        n9 = int(2500/16)
        n10 = int(1200/16)
        n11 = int(1200/16)
        n12 = int(1200/16)

        n13 = int(1200/16) #3969  # nb samples
        n14 = int(1000/16)
        n15 = int(1000/16)
        n16 = int(1000/16)
        n17 = int(1000/16)
        n18 = int(5120/16)
        # print(n1, n2, n3, n4, n5, n6)
    # print(len(val1a1), len(val1a2), len(val1a3), len(val1a4), len(val1a5), len(val1a6))

    indices = random.sample(range(0, len(val1a1)), n1)
    # print("indices", indices)
    # print(indices, n1, len(val1a1), len(val3a1))
    xs = np.array([val1a1[i] for i in indices])
    xt = np.array([val3a1[i] for i in indices])

    M = ot.dist(xs.reshape((n1, 1)), xt.reshape((n1, 1)))
    M /= M.max()

    a, b = np.ones((n1,)) / n1, np.ones((n1,)) / n1
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a1[i] = transportedsample[k]
        k += 1 
    val11 = np.reshape(val1a1, (64,3,7,7))
    aggregiatedmodel.conv1.weight.data = torch.from_numpy(val11).to(cuda3)


    #*****************************************************************************
    indices = random.sample(range(0, len(val1a2)), n2)
    # print("indices", indices)
    xs = np.array([val1a2[i] for i in indices])
    xt = np.array([val3a2[i] for i in indices])

    M = ot.dist(xs.reshape((n2, 1)), xt.reshape((n2, 1)))
    M /= M.max()

    a, b = np.ones((n2,)) / n2, np.ones((n2,)) / n2
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a2[i] = transportedsample[k]
        k += 1
    val11 = np.reshape(val1a2, (64,64,3,3))

    # aggregiatedmodel.layer1.0.conv1.weight.data = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.state_dict()['layer1.0.conv1.weight'] = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a3)), n3)
    # print("indices", indices)
    xs = np.array([val1a3[i] for i in indices])
    xt = np.array([val3a3[i] for i in indices])


    M = ot.dist(xs.reshape((n3, 1)), xt.reshape((n3, 1)))
    M /= M.max()

    a, b = np.ones((n3,)) / n3, np.ones((n3,)) / n3
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a3[i] = transportedsample[k]    
        k += 1

    val11 = np.reshape(val1a3, (64,64,3,3))

    # aggregiatedmodel.layer1.0.conv2.weight.data = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.state_dict()['layer1.0.conv2.weight'] = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a4)), n4)
    # print("indices", indices)
    xs = np.array([val1a4[i] for i in indices])
    xt = np.array([val3a4[i] for i in indices])

    M = ot.dist(xs.reshape((n4, 1)), xt.reshape((n4, 1)))
    M /= M.max()

    a, b = np.ones((n4,)) / n4, np.ones((n4,)) / n4
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a4[i] = transportedsample[k]  
        k += 1

    val11 = np.reshape(val1a4, (64,64,3,3))

    aggregiatedmodel.state_dict()['layer1.1.conv1.weight'] = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a5)), n5)
    # print("indices", indices)
    xs = np.array([val1a5[i] for i in indices])
    xt = np.array([val3a5[i] for i in indices])

    M = ot.dist(xs.reshape((n5, 1)), xt.reshape((n5, 1)))
    M /= M.max()

    a, b = np.ones((n5,)) / n5, np.ones((n5,)) / n5
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a5[i] = transportedsample[k]  
        k += 1

    val11 = np.reshape(val1a5, (64,64,3,3))

    aggregiatedmodel.state_dict()['layer1.1.conv2.weight'] = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************
    indices = random.sample(range(0, len(val1a6)), n6)
    # print("indices", indices)
    xs = np.array([val1a6[i] for i in indices])
    xt = np.array([val3a6[i] for i in indices])


    M = ot.dist(xs.reshape((n6, 1)), xt.reshape((n6, 1)))
    M /= M.max()

    a, b = np.ones((n6,)) / n6, np.ones((n6,)) / n6
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a6[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a6, (128,64,3,3))

    aggregiatedmodel.state_dict()['layer2.0.conv1.weight'] = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a7)), n7)
    # print("indices", indices)
    xs = np.array([val1a7[i] for i in indices])
    xt = np.array([val3a7[i] for i in indices])


    M = ot.dist(xs.reshape((n7, 1)), xt.reshape((n7, 1)))
    M /= M.max()

    a, b = np.ones((n7,)) / n7, np.ones((n7,)) / n7
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a7[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a7, (128,128,3,3))

    aggregiatedmodel.state_dict()['layer2.0.conv2.weight'] = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************
    indices = random.sample(range(0, len(val1a8)), n8)
    # print("indices", indices)
    xs = np.array([val1a8[i] for i in indices])
    xt = np.array([val3a8[i] for i in indices])


    M = ot.dist(xs.reshape((n8, 1)), xt.reshape((n8, 1)))
    M /= M.max()

    a, b = np.ones((n8,)) / n8, np.ones((n8,)) / n8
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a8[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a8, (128,128,3,3))

    aggregiatedmodel.state_dict()['layer2.1.conv1.weight'] = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************
    indices = random.sample(range(0, len(val1a9)), n9)
    # print("indices", indices)
    xs = np.array([val1a9[i] for i in indices])
    xt = np.array([val3a9[i] for i in indices])


    M = ot.dist(xs.reshape((n9, 1)), xt.reshape((n9, 1)))
    M /= M.max()

    a, b = np.ones((n9,)) / n9, np.ones((n9,)) / n9
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a9[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a9, (128,128,3,3))

    aggregiatedmodel.state_dict()['layer2.1.conv2.weight'] = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************

    indices = random.sample(range(0, len(val1a10)), n10)
    # print("indices", indices)
    xs = np.array([val1a10[i] for i in indices])
    xt = np.array([val3a10[i] for i in indices])


    M = ot.dist(xs.reshape((n10, 1)), xt.reshape((n10, 1)))
    M /= M.max()

    a, b = np.ones((n10,)) / n10, np.ones((n10,)) / n10
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a10[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a10, (256,128,3,3))

    aggregiatedmodel.state_dict()['layer3.0.conv1.weight'] = torch.from_numpy(val11).to(cuda3)
#*****************************************************************************

    indices = random.sample(range(0, len(val1a11)), n11)
    # print("indices", indices)
    xs = np.array([val1a11[i] for i in indices])
    xt = np.array([val3a11[i] for i in indices])


    M = ot.dist(xs.reshape((n11, 1)), xt.reshape((n11, 1)))
    M /= M.max()

    a, b = np.ones((n11,)) / n11, np.ones((n11,)) / n11
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a11[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a11, (256,256,3,3))

    aggregiatedmodel.state_dict()['layer3.0.conv2.weight'] = torch.from_numpy(val11).to(cuda3)
#*****************************************************************************
    indices = random.sample(range(0, len(val1a12)), n12)
    # print("indices", indices)
    xs = np.array([val1a12[i] for i in indices])
    xt = np.array([val3a12[i] for i in indices])


    M = ot.dist(xs.reshape((n12, 1)), xt.reshape((n12, 1)))
    M /= M.max()

    a, b = np.ones((n12,)) / n12, np.ones((n12,)) / n12
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a12[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a12, (256,256,3,3))

    aggregiatedmodel.state_dict()['layer3.1.conv1.weight'] = torch.from_numpy(val11).to(cuda3)
#*****************************************************************************

    indices = random.sample(range(0, len(val1a13)), n13)
    # print("indices", indices)
    xs = np.array([val1a13[i] for i in indices])
    xt = np.array([val3a13[i] for i in indices])


    M = ot.dist(xs.reshape((n13, 1)), xt.reshape((n13, 1)))
    M /= M.max()

    a, b = np.ones((n13,)) / n13, np.ones((n13,)) / n13
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a13[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a13, (256,256,3,3))

    aggregiatedmodel.state_dict()['layer3.1.conv2.weight'] = torch.from_numpy(val11).to(cuda3)
#*****************************************************************************

    indices = random.sample(range(0, len(val1a14)), n14)
    # print("indices", indices)
    xs = np.array([val1a14[i] for i in indices])
    xt = np.array([val3a14[i] for i in indices])


    M = ot.dist(xs.reshape((n14, 1)), xt.reshape((n14, 1)))
    M /= M.max()

    a, b = np.ones((n14,)) / n14, np.ones((n14,)) / n14
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a14[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a14, (512,256,3,3))

    aggregiatedmodel.state_dict()['layer4.0.conv1.weight'] = torch.from_numpy(val11).to(cuda3)
    
#*****************************************************************************

    indices = random.sample(range(0, len(val1a15)), n15)
    # print("indices", indices)
    xs = np.array([val1a15[i] for i in indices])
    xt = np.array([val3a15[i] for i in indices])


    M = ot.dist(xs.reshape((n15, 1)), xt.reshape((n15, 1)))
    M /= M.max()

    a, b = np.ones((n15,)) / n15, np.ones((n15,)) / n15
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a15[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a15, (512,512,3,3))

    aggregiatedmodel.state_dict()['layer4.0.conv2.weight'] = torch.from_numpy(val11).to(cuda3)
    
#*****************************************************************************

    indices = random.sample(range(0, len(val1a16)), n16)
    # print("indices", indices)
    xs = np.array([val1a16[i] for i in indices])
    xt = np.array([val3a16[i] for i in indices])


    M = ot.dist(xs.reshape((n16, 1)), xt.reshape((n16, 1)))
    M /= M.max()

    a, b = np.ones((n16,)) / n16, np.ones((n16,)) / n16
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a16[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a16, (512,512,3,3))

    aggregiatedmodel.state_dict()['layer4.1.conv1.weight'] = torch.from_numpy(val11).to(cuda3)
    
#*****************************************************************************

    indices = random.sample(range(0, len(val1a17)), n17)
    # print("indices", indices)
    xs = np.array([val1a17[i] for i in indices])
    xt = np.array([val3a17[i] for i in indices])


    M = ot.dist(xs.reshape((n17, 1)), xt.reshape((n17, 1)))
    M /= M.max()

    a, b = np.ones((n17,)) / n17, np.ones((n17,)) / n17
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a17[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a17, (512,512,3,3))

    aggregiatedmodel.state_dict()['layer4.1.conv2.weight'] = torch.from_numpy(val11).to(cuda3)
    
#*****************************************************************************

    indices = random.sample(range(0, len(val1a18)), n18)
    # print("indices", indices)
    xs = np.array([val1a18[i] for i in indices])
    xt = np.array([val3a18[i] for i in indices])


    M = ot.dist(xs.reshape((n18, 1)), xt.reshape((n18, 1)))
    M /= M.max()

    a, b = np.ones((n18,)) / n18, np.ones((n18,)) / n18
    G0 = ot.emd(a, b, M, numItermax=1000000)

    transportedsample = maps_t(G0, xt)
    for i in indices:
        k = 0
        val1a18[i] = transportedsample[k]  
        k += 1
    val11 = np.reshape(val1a18, (10,512))

    aggregiatedmodel.fc.weight.data = torch.from_numpy(val11).to(cuda3)
    
#*****************************************************************************

    return aggregiatedmodel

def otattackra(aggregiatedmodel):
    val1a1=[]
    val1a2=[]
    val1a3=[]
    val1a4=[]
    val1a5=[]
    val1a6=[]


    for name, para in aggregiatedmodel.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('fc1.weight'):
            val1a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv1.weight'):
            val1a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv2.weight'):
            val1a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv3.weight'):
            val1a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv4.weight'):
            val1a6 = torch.flatten(para).cpu().detach().numpy()

    model_path = "./GTSRB_25C_13A_het_40.pt"
    global_model3 = CustomCNN().to(cuda3)
    global_model3.load_state_dict(torch.load(model_path, map_location='cpu'))
    global_model3.eval()
 
    val3a1=[]
    val3a2=[]
    val3a3=[]
    val3a4=[]
    val3a5=[]
    val3a6=[]


    for name, para in global_model3.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('fc1.weight'):
            val3a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val3a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv1.weight'):
            val3a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv2.weight'):
            val3a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv3.weight'):
            val3a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('conv4.weight'):
            val3a6 = torch.flatten(para).cpu().detach().numpy()
    #****************************************************************************
    # n1 = 10 #3969  # nb samples
    # n2 = 10
    # n3 = 1000
    # n4 = 500
    # n5 = 250
    # n6 = 100

    AccOT = []
    Accdirect = [] 
    indices = random.sample(range(0, len(val1a1)), 5000)
    # print("indices", indices)
    xs = np.array([val1a1[i] for i in indices])
    xt = np.array([val3a1[i] for i in indices])

    for i in indices:
        val1a1[i] = val3a1[i]
    val111 = np.reshape(val1a1, (512,16384))
    aggregiatedmodel.fc1.weight.data = torch.from_numpy(val111).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a2)), 5000)
    # print("indices", indices)
    xs = np.array([val1a2[i] for i in indices])
    xt = np.array([val3a2[i] for i in indices])

    for i in indices:
        val1a2[i] = val3a2[i]
    val111 = np.reshape(val1a2, (43,512))
    aggregiatedmodel.fc2.weight.data = torch.from_numpy(val111).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a3)), 4800)
    # print("indices", indices)
    xs = np.array([val1a3[i] for i in indices])
    xt = np.array([val3a3[i] for i in indices])

    for i in indices:
        val1a3[i] = val3a3[i]
    val111 = np.reshape(val1a3, (64,3,5,5))
    aggregiatedmodel.conv1.weight.data = torch.from_numpy(val111).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a4)), 5000)
    # print("indices", indices)
    xs = np.array([val1a4[i] for i in indices])
    xt = np.array([val3a4[i] for i in indices])

    for i in indices:
        val1a4[i] = val3a4[i]
    val111 = np.reshape(val1a4, (128,64,3,3))
    aggregiatedmodel.conv2.weight.data = torch.from_numpy(val111).to(cuda3)


    #*****************************************************************************
    indices = random.sample(range(0, len(val1a5)), 5000)
    # print("indices", indices)
    xs = np.array([val1a5[i] for i in indices])
    xt = np.array([val3a5[i] for i in indices])

    for i in indices:
        val1a5[i] = val3a5[i]
    val111 = np.reshape(val1a5, (256,128,1,1))
    aggregiatedmodel.conv3.weight.data = torch.from_numpy(val111).to(cuda3)
    #*****************************************************************************
    indices = random.sample(range(0, len(val1a6)), 5000)
    # print("indices", indices)
    xs = np.array([val1a6[i] for i in indices])
    xt = np.array([val3a6[i] for i in indices])

    for i in indices:
        val1a6[i] = val3a6[i]
    val111 = np.reshape(val1a6, (256,256,1,1))
    aggregiatedmodel.conv4.weight.data = torch.from_numpy(val111).to(cuda3)
    #*****************************************************************************

#*****************************************************************************
    return aggregiatedmodel

#initialize global model
global_model = CustomCNN.to(cuda3)
global_model.train()

model_path = "./globalmodel_base_30C1A_heter_4102023.pt"
global_model3 = CustomCNN.to(cuda3)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# print(CustomCNN)

# for name, para in global_model.named_parameters():
#     print(name, len(torch.flatten(para).cpu().detach().numpy()), para.shape)


global_model3.load_state_dict(state_dict)
global_model3.eval()

global_acc, global_loss = test(global_model3, test_loader)

print(global_acc)


learning_rate = 0.01 
global_epochs = 100
local_epochs = 1

optimizer  = torch.optim.SGD(global_model.parameters(), lr=learning_rate, momentum=0.9)
criterion  = nn.CrossEntropyLoss()
        
#commence global training loop
GTA =[]
cs = []
for g_epoch in range(global_epochs):
            
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.parameters()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random_clients = random.sample(client_names, k=40)
    print(random_clients)
    # random.shuffle(client_names)
    
    #ToDO: a random fraction C of clients is selected and put in client_names_sel
    client_names_sel = random_clients
    # print(client_names_sel)
    
    client_models = []
    scaling_factor = []
    clients_data_list = []

    #loop through each client and create new local model
    for client in tqdm(client_names_sel):
        local_model = CustomCNN.to(cuda3)
        local_model.load_state_dict(global_model.state_dict())
        optimizer  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        loss_func  = nn.CrossEntropyLoss()
        local_model.train()
        for epoch in range(local_epochs):
          client_data_size = 0
          for batch_idx, (data, target) in enumerate(clients_batched[client]):
                client_data_size += len(target)
                # print('client_data_size ', client_data_size)
                data, target = data.to(cuda3), target.to(cuda3)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
                del loss
                torch.cuda.empty_cache()
                del output
                torch.cuda.empty_cache()
                del data
                torch.cuda.empty_cache()
                del target
                torch.cuda.empty_cache()
        if client in ["clients_1", "clients_3", "clients_5", "clients_7", "clients_9", "clients_11", "clients_13", "clients_15", "clients_17", "clients_19"]: 
            local_model = otattackfrombeginning(local_model, g_epoch)
            local_model.load_state_dict(global_model3.state_dict())

                    #Extract the weights of each layer in both models
            weights1 = []
            weights2 = []

            for layer in local_model.state_dict().values():
                weights1.append(layer.view(-1))
            for layer in global_model.state_dict().values():
                weights2.append(layer.view(-1))

            # Concatenate the weights of each layer in both models into a single 1D tensor
            weights1 = torch.cat(weights1)
            weights2 = torch.cat(weights2)

            # Calculate the cosine similarity between the two 1D tensors
            cos_sim = torch.nn.functional.cosine_similarity(weights1, weights2, dim=0)

            cs.append(cos_sim)
            # print("nclp-1A-cifar10-cosim",g_epoch, max(cs))
            np.save('csnclp1A-cifar10.npy', cs)

        client_models.append(local_model)
        clients_data_list.append(client_data_size)
        del local_model
        torch.cuda.empty_cache()
        
        # print('client_data_size ', clients_data_list)

    tot_data = sum(clients_data_list)
    scaling_factor = [client_data / tot_data for client_data in clients_data_list]
    # scaling_factor.append(1.0) 
    #to get the average over all the local model, we simply take the sum of the scaled weights
    client_models = scale_model_weights(client_models, scaling_factor)
    global_model = server_aggregate(global_model,client_models)
    # global_model = otattackfrombeginning(global_model, g_epoch)
    global_acc, global_loss = test(global_model, test_loader)
    # print('global_epochs: {} | global_loss: {} | global_accuracy: {}'.format(g_epoch+1, global_loss, global_acc))
    # torch.save(global_model.state_dict(), './GAmodels231221/FL_GTSRB_25C_Afrombeginning_homo_3Jan2022_'+str(g_epoch)+'.pt')
    
    GTA.append(global_acc)
    print("nclp-10A-cifar10-gta",g_epoch, max(GTA))
    np.save('gtanclp10A-cifar10.npy', GTA)



print('Federated learning process completed...ok')
print('--------------------------------')
