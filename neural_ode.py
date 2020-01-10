import time
import random, math
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

tol = 1e-3
nepochs = 400
lr = .01

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.linear1 = nn.Linear(2, dim)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(dim, dim)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(dim, 2)

    def forward(self, t, x):
        out = self.linear1(x)
        out = self.tanh1(out)
        out = self.linear2(out)
        out = self.tanh2(out)
        out = self.linear3(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out[1]

model = nn.Sequential(ODEBlock(ODEfunc(64)))

x_train = []
y_train = []
for i in range(5000):
    r = random.uniform(0, 1000)
    theta = random.uniform(-math.pi/2, math.pi/2)
    
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    
    x_train.append([r/1000, theta/10])
    y_train.append([x/1000, y/1000])
    
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for itr in range(nepochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = torch.mean(torch.abs(y_pred - y_train))
    loss.backward()
    optimizer.step()

    if itr % 20 == 0:
        with torch.no_grad():
            prediction = model(x_train)
            for i in range(10):
                print(prediction[i]*1000, y_train[i]*1000)
                print('% Error', abs((y_train[i] - prediction[i])/y_train[i]*100))
            print()