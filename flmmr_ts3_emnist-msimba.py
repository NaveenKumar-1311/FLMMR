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
    cuda3 = torch.device('cuda:1') 
else:  
    cuda3 = torch.device('cpu') 


train_path = './Train'
test_csv_file_path = './Test.csv'


# In[5]:


NUM_CLASSES = 62
IMG_SIZE = 32
BATCH_SIZE = 64


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

train_sampler1, test_sampler = torch.utils.data.random_split(train_sampler, (round(0.99*len(train_sampler)), round(0.01*len(train_sampler))))

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
        
    # for i in range(len(shards)):
    #     print('client {}: data size: {}'.format(client_names[i], len(shards[i])))
        
    return {client_names[i]: shards[i] for i in range(len(client_names))}

clients = create_clients(train_sampler, num_clients=10000,initial='clients', alpha=1)
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


#Defining the convolutional neural network
#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# CustomCNN = models.resnet18()

# #print(CustomCNN)

# # print(CustomCNN.fc.out_features) # 1000 



# # Freeze training for all layers
# # for param in CustomCNN.features.parameters():
# #     param.require_grad = False

# # Newly created modules have require_grad=True by default
# num_features = CustomCNN.fc.in_features
# #features = list(CustomCNN.fc.children())[:-1] # Remove last layer
# #features.extend([nn.Linear(num_features, NUM_CLASSES)]) # Add our layer with 4 outputs
# #CustomCNN.fc = nn.Sequential(*features) # Replace the model fc
# CustomCNN.fc = nn.Linear(num_features, NUM_CLASSES)

global_model = LeNet5(NUM_CLASSES).to(cuda3)

# print(global_model)

for name, para in global_model.named_parameters():
    print(name, len(torch.flatten(para).cpu().detach().numpy()), para.shape)

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
    


    for name, para in aggregiatedmodel.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.0.weight'):
            val1a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.weight'):
            val1a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.weight'):
            val1a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val1a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc1.weight'):
            val1a6 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val1a7 = torch.flatten(para).cpu().detach().numpy()
        

    model_path = "./surrogate_bad_emnist.pt"
    global_model3 = LeNet5(NUM_CLASSES).to(cuda3)
    global_model3.load_state_dict(torch.load(model_path, map_location='cpu'))
    global_model3.eval()
 
    val3a1=[]
    val3a2=[]
    val3a3=[]
    val3a4=[]
    val3a5=[]
    val3a6=[]
    val3a7=[]
 


    for name, para in global_model3.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.0.weight'):
            val3a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.weight'):
            val3a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.weight'):
            val3a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.weight'):
            val3a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val3a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc1.weight'):
            val3a6 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val3a7 = torch.flatten(para).cpu().detach().numpy()
 
    #****************************************************************************

    if g_epoch >= 0 and g_epoch <=20:
        # print("Entering CLP --> 1")
        n1 = int(100) #3969  # nb samples
        n2 = int(3)
        n3 = int(120)
        n4 = int(8)
        n5 = int(600)
        n6 = int(120)

        n7 = int(120) #3969  # nb samples
 
 

        # print(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18)
    elif g_epoch > 20 and g_epoch <=40:
        # print("Entering CLP --> 2")
        n1 = int(50) #3969  # nb samples
        n2 = int(3)
        n3 = int(60)
        n4 = int(8)
        n5 = int(300)
        n6 = int(60)

        n7 = int(60) #3969  # nb samples
 
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 40 and g_epoch <=60:
        # print("Entering CLP --> 3")
        n1 = int(50) #3969  # nb samples
        n2 = int(3)
        n3 = int(60)
        n4 = int(8)
        n5 = int(300)
        n6 = int(60)

        n7 = int(60) #3969  # nb samples
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 60 and g_epoch <=80:
        # print("Entering CLP --> 4")
        n1 = int(25) #3969  # nb samples
        n2 = int(3)
        n3 = int(30)
        n4 = int(8)
        n5 = int(150)
        n6 = int(30)

        n7 = int(30) #3969  # nb samples
        # print(n1, n2, n3, n4, n5, n6)
    else:
        # print("Entering CLP --> 5")
        n1 = int(25) #3969  # nb samples
        n2 = int(3)
        n3 = int(30)
        n4 = int(8)
        n5 = int(150)
        n6 = int(30)

        n7 = int(30) #3969  # nb samples
 
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
    val11 = np.reshape(val1a1, (6,3,5,5))
    aggregiatedmodel.layer1[0].weight.data = torch.from_numpy(val11).to(cuda3)

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
    val11 = np.reshape(val1a2, (6))
    # print("val11:",val11)
    # aggregiatedmodel.layer1.0.conv1.weight.data = torch.from_numpy(val11).to(cuda3)
    # aggregiatedmodel.state_dict()['layer1.1.weight'] = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.layer1[1].weight.data = torch.from_numpy(val11).to(cuda3)
    for name, para in aggregiatedmodel.named_parameters():
        # print('{}: {} : {}'.format(name, para.shape, para))    
        # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.1.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
    # print("aggregated:",aggregiatedmodel.layer1[1].weight.data)
    # print("val3a2:",val3a2)
    # print("val1a2:",val1a2)
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

    val11 = np.reshape(val1a3, (16,6,5,5))

    # aggregiatedmodel.layer1.0.conv2.weight.data = torch.from_numpy(val11).to(cuda3)
    # aggregiatedmodel.state_dict()['layer2.0.weight'] = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.layer2[0].weight.data = torch.from_numpy(val11).to(cuda3)
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

    val11 = np.reshape(val1a4, (16))

    aggregiatedmodel.layer2[1].weight.data = torch.from_numpy(val11).to(cuda3)

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

    val11 = np.reshape(val1a5, (120,400))

    aggregiatedmodel.fc.weight.data = torch.from_numpy(val11).to(cuda3)

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
    val11 = np.reshape(val1a6, (84,120))

    aggregiatedmodel.fc1.weight.data = torch.from_numpy(val11).to(cuda3)

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
    val11 = np.reshape(val1a7, (62,84))

    aggregiatedmodel.fc2.weight.data = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************

    return aggregiatedmodel

def otattackra(aggregiatedmodel, g_epoch):
    val1a1=[]
    val1a2=[]
    val1a3=[]
    val1a4=[]
    val1a5=[]
    val1a6=[]
    val1a7=[]
    


    for name, para in aggregiatedmodel.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.0.weight'):
            val1a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.weight'):
            val1a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.weight'):
            val1a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val1a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc1.weight'):
            val1a6 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val1a7 = torch.flatten(para).cpu().detach().numpy()
        

    model_path = "./surrogate_bad_emnist.pt"
    global_model3 = LeNet5(NUM_CLASSES).to(cuda3)
    global_model3.load_state_dict(torch.load(model_path, map_location='cpu'))
    global_model3.eval()
 
    val3a1=[]
    val3a2=[]
    val3a3=[]
    val3a4=[]
    val3a5=[]
    val3a6=[]
    val3a7=[]
 


    for name, para in global_model3.named_parameters():
            # print('{}: {} : {}'.format(name, para.shape, para))    
            # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.0.weight'):
            val3a1 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer1.1.weight'):
            val3a2 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.0.weight'):
            val3a3 = torch.flatten(para).cpu().detach().numpy()
        if name == str('layer2.1.weight'):
            val3a4 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc.weight'):
            val3a5 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc1.weight'):
            val3a6 = torch.flatten(para).cpu().detach().numpy()
        if name == str('fc2.weight'):
            val3a7 = torch.flatten(para).cpu().detach().numpy()
 
    #****************************************************************************

    if g_epoch >= 0 and g_epoch <=20:
        # print("Entering CLP --> 1")
        n1 = int(450) #3969  # nb samples
        n2 = int(3)
        n3 = int(1200)
        n4 = int(8)
        n5 = int(5000)
        n6 = int(5000)

        n7 = int(5208) #3969  # nb samples
 
 

        # print(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18)
    elif g_epoch > 20 and g_epoch <=40:
        # print("Entering CLP --> 2")
        n1 = int(100) #3969  # nb samples
        n2 = int(3)
        n3 = int(120)
        n4 = int(8)
        n5 = int(600)
        n6 = int(120)

        n7 = int(120) #3969  # nb samples
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 40 and g_epoch <=60:
        # print("Entering CLP --> 3")
        n1 = int(50) #3969  # nb samples
        n2 = int(3)
        n3 = int(60)
        n4 = int(8)
        n5 = int(300)
        n6 = int(60)

        n7 = int(60) #3969  # nb samples
        # print(n1, n2, n3, n4, n5, n6)
    elif g_epoch > 60 and g_epoch <=80:
        # print("Entering CLP --> 4")
        n1 = int(50) #3969  # nb samples
        n2 = int(3)
        n3 = int(60)
        n4 = int(8)
        n5 = int(300)
        n6 = int(60)

        n7 = int(60) #3969  # nb samples
        # print(n1, n2, n3, n4, n5, n6)
    else:
        # print("Entering CLP --> 5")
        n1 = int(25) #3969  # nb samples
        n2 = int(3)
        n3 = int(30)
        n4 = int(8)
        n5 = int(150)
        n6 = int(30)

        n7 = int(30) #3969  # nb samples
 
        # print(n1, n2, n3, n4, n5, n6)
    # print(len(val1a1), len(val1a2), len(val1a3), len(val1a4), len(val1a5), len(val1a6))

    indices = random.sample(range(0, len(val1a1)), n1)
    # print("indices", indices)
    # print(indices, n1, len(val1a1), len(val3a1))
    xs = np.array([val1a1[i] for i in indices])
    xt = np.array([val3a1[i] for i in indices])

    for i in indices:
        val1a1[i] = val3a1[i]
    val11 = np.reshape(val1a1, (6,3,5,5))
    aggregiatedmodel.layer1[0].weight.data = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a2)), n2)
    # print("indices", indices)
    for i in indices:
        val1a2[i] = val3a2[i]
    val11 = np.reshape(val1a2, (6))
    # print("val11:",val11)
    # aggregiatedmodel.layer1.0.conv1.weight.data = torch.from_numpy(val11).to(cuda3)
    # aggregiatedmodel.state_dict()['layer1.1.weight'] = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.layer1[1].weight.data = torch.from_numpy(val11).to(cuda3)
    for name, para in aggregiatedmodel.named_parameters():
        # print('{}: {} : {}'.format(name, para.shape, para))    
        # print('{}: {}'.format(name, para.shape)) 
        if name == str('layer1.1.weight'):
            val1a2 = torch.flatten(para).cpu().detach().numpy()
    # print("aggregated:",aggregiatedmodel.layer1[1].weight.data)
    # print("val3a2:",val3a2)
    # print("val1a2:",val1a2)
    #*****************************************************************************
    indices = random.sample(range(0, len(val1a3)), n3)
    # print("indices", indices)
    for i in indices:
        val1a3[i] = val3a3[i]

    val11 = np.reshape(val1a3, (16,6,5,5))

    # aggregiatedmodel.layer1.0.conv2.weight.data = torch.from_numpy(val11).to(cuda3)
    # aggregiatedmodel.state_dict()['layer2.0.weight'] = torch.from_numpy(val11).to(cuda3)
    aggregiatedmodel.layer2[0].weight.data = torch.from_numpy(val11).to(cuda3)
    #*****************************************************************************
    indices = random.sample(range(0, len(val1a4)), n4)
    # print("indices", indices)
    for i in indices:
        val1a4[i] = val3a4[i]

    val11 = np.reshape(val1a4, (16))

    aggregiatedmodel.layer2[1].weight.data = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a5)), n5)
    # print("indices", indices)
    for i in indices:
        val1a5[i] = val3a5[i]

    val11 = np.reshape(val1a5, (120,400))

    aggregiatedmodel.fc.weight.data = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************
    indices = random.sample(range(0, len(val1a6)), n6)
    # print("indices", indices)
    for i in indices:
        val1a6[i] = val3a6[i]
    val11 = np.reshape(val1a6, (84,120))

    aggregiatedmodel.fc1.weight.data = torch.from_numpy(val11).to(cuda3)

    #*****************************************************************************
    indices = random.sample(range(0, len(val1a7)), n7)
    # print("indices", indices)
    for i in indices:
        val1a7[i] = val3a7[i]
    val11 = np.reshape(val1a7, (62,84))

    aggregiatedmodel.fc2.weight.data = torch.from_numpy(val11).to(cuda3)

#*****************************************************************************


# #*****************************************************************************
    return aggregiatedmodel

#initialize global model
# global_model = CustomCNN.to(cuda3)
global_model.train()

model_path = "./surrogate_bad_emnist.pt"
global_model3 = LeNet5(NUM_CLASSES).to(cuda3)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# print(CustomCNN)

# for name, para in global_model.named_parameters():
#     print(name, len(torch.flatten(para).cpu().detach().numpy()), para.shape)


global_model3.load_state_dict(state_dict)
global_model3.eval()

global_acc, global_loss = test(global_model3, test_loader)

print(global_acc)

def get_probs(model, x, confused_class):
    probs = (model(x))[:, confused_class]
    return torch.diag(probs.data)

def MSimBA(model, x, y, num_iters, epsilon):
    
    x_org = x.clone()
    n_dims = x.reshape(1, -1).size(1)
    perm = torch.randperm(n_dims)
    org_probs = model(x)
    confused_class = torch.topk(org_probs.squeeze(), 2, dim=0, largest=True, sorted=True).indices[1]
    confused_prob = org_prob[0, confused_class]
    last_prob = get_probs(model, x_org, confused_class)
    new_class = y.clone()
    i = 0
    k = 1

    while ((i < num_iters) and ((y == new_class) or (torch.abs((output)[:, y] - (output)[:, new_class]) <= 0.1))):
        # print(i)
        diff = torch.zeros(n_dims).to(cuda3)
        diff[perm[i % len(perm)]] = epsilon
        perturbation = diff.reshape(x.size())

        left_prob = get_probs(model, (x - perturbation), confused_class)
        
        if left_prob > last_prob:
            x = (x - perturbation)
            last_prob = left_prob
    
        else:
            right_prob = get_probs(model, (x + perturbation).to(cuda3), confused_class)
            if right_prob > last_prob:
                x = (x + perturbation)
                last_prob = right_prob

        output = model(x)
        new_class = torch.argmax(output)

        i += 1
    return x, model(x), i


learning_rate = 0.01 
global_epochs = 200
local_epochs = 5

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
    random_clients = random.sample(client_names, k=100)
    # print(random_clients)
    # random.shuffle(client_names)
    
    #ToDO: a random fraction C of clients is selected and put in client_names_sel
    client_names_sel = random_clients
    # print(client_names_sel)
    
    client_models = []
    scaling_factor = []
    clients_data_list = []

    #loop through each client and create new local model
    for client in tqdm(client_names_sel):
        local_model = LeNet5(NUM_CLASSES).to(cuda3)
        local_model.load_state_dict(global_model.state_dict())
        optimizer  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
        loss_func  = nn.CrossEntropyLoss()
        local_model.train()
        for epoch in range(local_epochs):
          client_data_size = 0
          for batch_idx, (data, target) in enumerate(clients_batched[client]):
            client_data_size += len(target)
            if client == "clients_1":
                org_img = data.clone()
                org_label = target.clone()
                data = torch.zeros(64, 3, 32, 32)
                target = torch.zeros(64)
                org_prob = local_model(data.to(cuda3))
                org_class = torch.argmax(org_prob, dim=1)
                for j in range (0, 64):
                    # print("Attacking client", batch_idx)
                    data[j], adv_prob, itera = MSimBA(local_model.to(cuda3), org_img[j].clone().to(cuda3).unsqueeze(0), org_label[j].clone().to(cuda3).unsqueeze(0), 100, 0.7)
                    target[j] = torch.argmax(adv_prob)
            data, target = data.to(cuda3), target.to(cuda3)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            del loss
            # torch.cuda.empty_cache()
            del output
            # torch.cuda.empty_cache()
            del data
            # torch.cuda.empty_cache()
            del target
                # torch.cuda.empty_cache()
        # if client == "clients_1": 
        #     local_model = otattackra(local_model, g_epoch)

        client_models.append(local_model)

        clients_data_list.append(client_data_size)
        del local_model
        # torch.cuda.empty_cache()
        
        # print('client_data_size ', clients_data_list)

    tot_data = sum(clients_data_list)
    scaling_factor = [client_data / tot_data for client_data in clients_data_list]
    # scaling_factor.append(1.0) 
    #to get the average over all the local model, we simply take the sum of the scaled weights
    client_models = scale_model_weights(client_models, scaling_factor)
    global_model = server_aggregate(global_model,client_models)
    # global_model = otattackra(global_model, g_epoch)
    global_acc, global_loss = test(global_model, test_loader)

    
    GTA.append(global_acc)
    print("gtanclp-ts3-emnist-msimba",g_epoch, max(GTA))
    np.save('gtanclp-ts3-emnist-msimba.npy', GTA)
    # torch.save(global_model.state_dict(),'surrogate_bad_emnist.pt')


print('Federated learning process completed...ok')
print('--------------------------------')
