#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRU_Model(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,num_class):
        super(GRU_Model,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.GRU = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self,x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        print h0.size()
        out, _ = self.GRU(x,h0)
        print out.size()
        out = self.fc(out[:,-1,:])
        print out.size()
        return out


sequence_length = 10
input_size = 1530
hidden_size = 128
num_layers = 2
num_classes = 30
batch_size = 100
num_epochs = 2
learning_rate = 0.01

rnn = GRU_Model(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

images = Variable(torch.randn(8, sequence_length, input_size)).cuda()
outputs = rnn(images)
