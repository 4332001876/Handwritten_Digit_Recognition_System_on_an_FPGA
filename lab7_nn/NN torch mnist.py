import math
import numpy as np
import csv,pprint
import random
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F


from torchvision.datasets import mnist
from torchvision import datasets,transforms
#from mnist.config import parser

#train_set=mnist.MNIST('./data',train=True,download=True)
#test_set=mnist.MNIST('./data',train=False,download=True)


train_set=datasets.MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

test_set=datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)  #训练集
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)#测试集
val_loader=test_loader




class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self, x):
        x = x.view(-1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


nn=Net()
nn_state_dict = torch.load("tmp/model.txt")
nn.load_state_dict(nn_state_dict)  

optimizer = optim.Adam(nn.parameters(), lr=0.001)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=50):
    aaaaa=0
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
        
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            if aaaaa==0:
                with open("nn_input.txt", mode='w') as f:
                    f.write(inputs[1].detach().numpy().tolist().__str__())
                print(targets[1].detach().numpy().tolist())
                aaaaa+=1
            
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            #inputs = inputs.to(device)
            output = model(inputs)
            #targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))


train(nn,optimizer, F.cross_entropy, train_loader, val_loader,epochs=3)



torch.save(nn.state_dict(),'tmp/model.txt')    
'''
nn = SimpleNet()
nn_state_dict = torch.load("tmp/model.txt")
nn.load_state_dict(nn_state_dict)   
'''




















