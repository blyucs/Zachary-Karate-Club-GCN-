#!/usr/bin/env python
# coding: utf-8

# # Intro to Graph Convolutional Network

# In[1]:


import numpy as np


#  # PyTorch implementation

# In[13]:


from collections import namedtuple
from networkx import read_edgelist, set_node_attributes, to_numpy_matrix
from pandas import read_csv, Series
from numpy import array
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from graph_builder import build_karate_club_graph
import torch.nn.functional as F
# # Data preprocessing 

# In[14]:


DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'network']
)

def load_karate_club():
    network = read_edgelist(
        'karate.edgelist',
        nodetype=int)

    attributes = read_csv(
        'karate.attributes.csv',
        index_col=['node'])

    for attribute in attributes.columns.values:
        set_node_attributes(
            network,
            values=Series(
                attributes[attribute],
                index=attributes.index).to_dict(),
            name=attribute
        )
  
    X_train,y_train = map(array, zip(*[
        ([node], data['role'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ]))
  
    X_test, y_test = map(array, zip(*[
        ([node], data['community'] == 'Administrator')
        for node, data in network.nodes(data=True)
        if data['role'] == 'Member'
    ]))
 
    return DataSet(
        X_train, y_train,
        X_test, y_test,
        network)


# In[15]:


zkc = load_karate_club()
X_train_flattened = torch.flatten(torch.from_numpy(zkc.X_train))
X_test_flattened = torch.flatten(torch.from_numpy(zkc.X_test))
y_train = torch.from_numpy(zkc.y_train).to(torch.long)

print(y_train)
A = to_numpy_matrix(zkc.network)
A = torch.from_numpy(np.array(A))
print(A)
print(X_train_flattened)


# # SpectralRule and LogisticRegressor Modules

# In[16]:


class SpectralRule(nn.Module):
    
    def __init__(self,A,input_units,output_units,activation = 'tanh'):
        
        super(SpectralRule,self).__init__()
        
        self.input_units = input_units
        self.output_units = output_units
        self.linear_layer = nn.Linear(self.input_units,self.output_units)
        nn.init.xavier_normal_(self.linear_layer.weight)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        #Created Identity Matrix
        I = torch.eye(A.shape[1])
       
        #Adding self loops to the adjacency matrix
        A_hat = A + I
        A_hat = A_hat.to(torch.double)
       
        #Inverse degree Matrix
        D = torch.diag(torch.pow(torch.sum(A_hat,dim = 0),-0.5),0)
       
        #applying spectral rule
        self.A_hat = torch.matmul(torch.matmul(D,A_hat),D)
        self.A_hat.requires_grad = False
       
        
    def forward(self,X):
        
        #aggregation
        aggregation = torch.matmul(self.A_hat,X)
         
        #propagation
        linear_output = self.linear_layer(aggregation.to(torch.float))
      
        propagation = self.activation(linear_output)
          
        return propagation.to(torch.double)

class LogisticRegressor(nn.Module):
    
    def __init__(self,input_units,output_units):
        super(LogisticRegressor,self).__init__()
        
        self.Linear = nn.Linear(input_units,output_units,bias=True)
        nn.init.xavier_normal_(self.Linear.weight)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):
        linear_output = self.Linear(X.to(torch.float))
        # return self.sigmoid(linear_output)
        return linear_output


# # Building the Base Model

# In[17]:

identity = torch.eye(A.shape[1])
identity = identity.to(torch.double)
identity.requires_grad = False

# In[18]:

# hidden_layer_config = [(4,'tanh'), (8, 'tanh'), (4, 'tanh'), (2,'tanh')]
# hidden_layer_config = [(4,'tanh'), (4, 'tanh'), (2,'tanh')]
hidden_layer_config = [(4,'tanh'), (2,'tanh')]

class FeatureModel(nn.Module):
    def __init__(self,A,hidden_layer_config,initial_input_size):
        super(FeatureModel,self).__init__()
        
        self.hidden_layer_config = hidden_layer_config
        
        self.moduleList = list()
        
        self.initial_input_size = initial_input_size
        
        for input_size,activation in hidden_layer_config:
            
            self.moduleList.append(SpectralRule(A,self.initial_input_size,input_size,activation))
            self.initial_input_size = input_size
        
        self.sequentialModule = nn.Sequential(*self.moduleList)
      
    def forward(self,X):
        output = self.sequentialModule(X)
       
        return output

class ClassifierModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(ClassifierModel,self).__init__()
        self.logisticRegressor = LogisticRegressor(input_units=input_size,output_units= output_size)
        
    def forward(self,X):
        classified  = self.logisticRegressor(X)
        return classified
    

class HybridModel(nn.Module):
    def __init__(self,A,hidden_layer_config,initial_input_size):
        super(HybridModel,self).__init__()
        self.featureModel = FeatureModel(A,hidden_layer_config,identity.shape[1])
        self.featureModelOutputSize = self.featureModel.initial_input_size
        self.classifier = ClassifierModel(self.featureModelOutputSize, 2)
        self.featureModelOutput = None
    
    def forward(self,X):
        
        outputFeature = self.featureModel(X)
        classified = self.classifier(outputFeature)
        self.featureModelOutput = outputFeature
        return classified

# # Identity Matrix as feature

# In[19]:

model = HybridModel(A,hidden_layer_config,identity.shape[1])

output = model(identity)

zkc.y_test

output


label_nodes = torch.tensor([0, 33])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum=0.9)
featureoutput = None
all_logits = []
def train(model , epoch , criterion , optimizer , feature):
    plt.figure()
    cumLoss = 0
    losses = list()
    
    for j in range(epoch):
        two_loss = 0
        tmp_logits = []
        # for i,node in enumerate(X_train_flattened):
        cur_logits = model(feature)
        # output = cur_logits[node]
        # tmp_logits.append(output)
        # ground_truth = torch.reshape(y_train,output.shape)
        logp = F.log_softmax(cur_logits, 1)
        optimizer.zero_grad()

        loss = F.nll_loss(logp[label_nodes], y_train)
        # loss = criterion(output,ground_truth)
        #\print("loss: ",loss.data)
        # two_loss += loss.item()

        loss.backward()

        optimizer.step()

        # losses.append(loss)
        cumLoss += loss
        # all_logits.append(tmp_logits)
    print('avg loss: ',cumLoss/epoch)
    torch.save(model.state_dict(),"./gcn.pth")
    plt.plot(losses)
    plt.show()
    
train(model, 10000,criterion,optimizer,identity)


after = None
def test(features):
    
    model = HybridModel(A,hidden_layer_config,identity.shape[1])
    model.load_state_dict(torch.load("./gcn.pth"))
    model.eval()
    correct = 0 
    masked_output = list()
    # for i ,node in enumerate(X_test_flattened):
    output = model(features)
    masked_output.append(output.ge(0.5))
    
    return masked_output  
        

# In[27]:

masked = test(identity)
masked = [i.item() for i in masked]
masked

# In[28]:

test_gt = torch.from_numpy(zkc.y_test)
test_gt = [ i.item() for i in test_gt]
test_gt
counter = 0
tp = 0
fp = 0
fn = 0
tn = 0

correct = zip(masked,test_gt)
for (masked,gt) in list(correct):
    if masked == gt and masked == True:
        tp += 1
    if masked == gt and masked == False:
        tn += 1
    if masked == False and gt == True:
        fn += 1
    if masked == True and gt == False:
        fp += 1
accuracy = (tp + tn) / (tp+fp+fn+tn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print('accuracy ',accuracy)
print('precision ',precision)
print('recall ',recall)




# In[ ]:




