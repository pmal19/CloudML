import re
import os
import sys
import pdb
import time
import math
import socket
import random
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from itertools import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms, utils

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

from dataparser import *
from batcher import *
from readEmbeddings import *
from datasets import *
from models import *


class BiLSTMSentiment(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
		super(BiLSTMSentiment, self).__init__()
		self.hidden_dim = hidden_dim
		self.use_gpu = use_gpu
		self.batch_size = batch_size
		self.dropout = dropout
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
		self.hidden2label = nn.Linear(hidden_dim*2, label_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		if self.use_gpu:
			return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()), Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
		else:
			return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)), Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

	def forward(self, sentence):
		x = sentence.view(len(sentence), self.batch_size, -1)
		lstm_out, _ = self.lstm(x, self.hidden)
		y = self.hidden2label(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs


class sstDataset(Dataset):
	def __init__(self, sstPath, glovePath, transform = None):
		self.data = load_sst_data(sstPath)
		self.paddingElement = ['<s>']
		self.maxSentenceLength = self.maxlength(self.data)
		self.vocab = glove2dict(glovePath)

	def __getitem__(self, index):
		s = self.pad(self.data[index]['sentence_1'].split())
		s = self.embed(s)
		label = int(self.data[index]['label'])
		return (s), label

	def __len__(self):
		return len(self.data)

	def maxlength(self, data):
		maxSentenceLength = max([len(d['sentence_1'].split()) for d in data])
		# print("Max sentence length - ", maxSentenceLength)
		return maxSentenceLength

	def pad(self, sentence):
		return sentence + (51-len(sentence))*self.paddingElement

	def embed(self, sentence):
		vector = []
		for word in sentence:
			if str(word) in self.vocab:
				vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
			else:
				vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
		return vector


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
	print("Epoch start - {}".format(epoch))
	for batch_idx, (data, target) in enumerate(trainLoader):
		s1 = data.float()
		batch, _ = s1.shape
		if batchSize != batch:
			break
		s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,batch).transpose(1,2)
		if(use_cuda):
			s1, target = Variable(s1.cuda()), Variable(target.cuda())
		else:
			s1, target = Variable(s1), Variable(target)

		model.zero_grad()
		output = model(s1)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx == break_val:
			return
		if batch_idx % 100 == 0:
			dev_loss = 0
			n_correct = 0
			n_total = 0
			for idx, (dev_data, dev_target) in enumerate(devLoader):
				sd = dev_data.float()
				devbatchSize, _ = sd.shape
				if batchSize != devbatchSize:
					break
				sd = sd.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
				if(use_cuda):
					sd, dev_target = Variable(sd.cuda()), Variable(dev_target.cuda())
				else:
					sd, dev_target = Variable(sd), Variable(dev_target)
				dev_output = model(sd)
				dev_loss += criterion(dev_output, dev_target)
				n_correct += (torch.max(dev_output, 1)[1].view(dev_target.size()) == dev_target).sum()
				n_total += devbatchSize
			dev_acc = (100. * n_correct.data[0])/n_total
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDev Loss: {:.6f}\tDev Acc: {:.6f}'.format(epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.data, dev_loss.data, dev_acc))
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.item()))
	print('Epoch {} avg loss : {}'.format(epoch, epoch_loss/num_batches))
	return loss

def train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
	for epoch in range(numEpochs):
		loss = trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize, use_cuda, devLoader, devbatchSize)
		for idx, (dev_data, dev_target) in enumerate(devLoader):
			sd = dev_data.float()
			devbatchSize, _ = sd.shape
			if batchSize != devbatchSize:
				break
			sd = sd.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
			if(use_cuda):
				sd, dev_target = Variable(sd.cuda(device_id)), Variable(dev_target.cuda(device_id))
			else:
				sd, dev_target = Variable(sd), Variable(dev_target)
			dev_output = model(sd)
			dev_loss += criterion(dev_output, dev_target)
			n_correct += (torch.max(dev_output, 1)[1].view(dev_target.size()) == dev_target).sum()
			n_total += devbatchSize
		dev_acc = (100. * n_correct.data)/n_total
		print('Epoch: {} \tDev Loss: {:.6f}\tDev Acc: {:.6f}'.format(epoch, dev_loss.data, dev_acc))

sstPathTrain = "../Data/SST/trees/train.txt"
sstPathDev = "../Data/SST/trees/dev.txt"
glovePath = '../Data/glove.6B/glove.6B.100d.txt'

batchSize = 16
learningRate = 0.0001
momentum = 0.9
numWorkers = 0

numEpochs = 1

inp_dim = 100
model_dim = 100
num_layers = 2
reverse = False
bidirectional = True
dropout = 0.4

mlp_input_dim = 200
mlp_dim = 100
num_classes = 5
num_mlp_layers = 5
mlp_ln = True
classifier_dropout_rate = 0.4

training = True

use_cuda = torch.cuda.is_available()

if(use_cuda):
	the_gpu.gpu = 0

t1 = time.time()
trainingDataset = sstDataset(sstPathTrain, glovePath)
devDataset = sstDataset(sstPathDev, glovePath)
print('Time taken - {}'.format(time.time()-t1))
devbatchSize = batchSize

trainLoader = DataLoader(trainingDataset, batchSize, shuffle=False, num_workers = numWorkers)
devLoader = DataLoader(devDataset, devbatchSize, shuffle=False, num_workers = numWorkers)

model = BiLSTMSentiment(100, 100, 100, 5, use_cuda, batchSize) 
if(use_cuda):
	model.cuda()
if(use_cuda):
	criterion = nn.CrossEntropyLoss().cuda()
else:
	criterion = nn.CrossEntropyLoss()
criterion=nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

start_time = time.monotonic()
train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, batchSize)
end_time = time.monotonic()
print("Avg Time: {}".format(float(end_time-start_time)/numEpochs))
