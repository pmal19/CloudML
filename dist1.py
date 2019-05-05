import time
import sys
import os
import re
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms, utils
from itertools import *

from dataparser import *
from batcher import *
from readEmbeddings import *
from datasets import *
from models import *

import pdb

class ClassLSTM(nn.Module):
    """docstring for ClassLSTM"""
    def __init__(self, input_size, hidden_size, num_layers, batch, bias = True, batch_first = False, dropout = 0, bidirectional = False):
        super(ClassLSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias = True, batch_first = False, dropout = dropout, bidirectional = False)
        self.h0 = Variable(torch.randn(num_layers * self.num_directions, batch, hidden_size))
        self.c0 = Variable(torch.randn(num_layers * self.num_directions, batch, hidden_size))
    def forward(self, s1):
        output, hn = self.lstm(s1, (self.h0, self.c0))
        return output

class sstNet(nn.Module):
    """docstring for sstNet"""
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training, batchSize):
        super(sstNet, self).__init__()
        self.encoderSst = ClassLSTM(inp_dim, model_dim, num_layers, batchSize, bidirectional = bidirectional, dropout = dropout)
        self.classifierSst = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)

    def forward(self, s1):
        oE = self.encoderSst(s1)
        features = oE[-1]
        output = F.log_softmax(self.classifierSst(features))
        return output

    def encode(self, s):
        emb = self.encoderSst(s)
        return emb


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
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

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
        print("Max sentence length - ", maxSentenceLength)
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


class Partition(Dataset):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(Dataset):
    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(sstPath, glovePath, batchSize, transformations=None):
    dataset = sstDataset(sstPath, glovePath, transformations)
    size = dist.get_world_size()
    bsz = batchSize
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=1)
    return train_set, bsz


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def run(rank, size, model, optimizer, criterion, epochs, trainLoader, bsz, devLoader, use_cuda):    
    torch.manual_seed(1234)
    epoch_loss = 0.0
    numberOfSamples = 0
    num_batches = ceil(len(trainLoader.dataset) / float(bsz))
    start_time = time.monotonic()
    for epoch in range(epochs):
        epoch_loss = 0.0
        numberOfSamples = 0
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

            numberOfSamples += data.size()[0]
            model.zero_grad()
            optimizer.zero_grad()
            output = model(s1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            if batch_idx == break_val:
                return
            if batch_idx % 1000 == 0:
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

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDev Loss: {:.6f}\tDev Acc: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainLoader.dataset),
                    100. * batch_idx / len(trainLoader), loss.data[0], dev_loss.data[0], dev_acc))

            # numberOfSamples += data.size()[0]
            # data, target = Variable(data), Variable(target)
            # optimizer.zero_grad()
            # output = model(data)
            # loss = criterion(output, target)
            # epoch_loss += loss.item()
            # loss.backward()
            # average_gradients(model)
            # optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.item()))
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
    end_time = time.monotonic()
    average_time = torch.Tensor([(end_time - start_time)/epochs])
    weighted_loss = torch.Tensor([(epoch_loss/num_batches) * numberOfSamples])
    numberOfSamples = torch.Tensor([numberOfSamples])
    dist.all_reduce(weighted_loss, op=dist.reduce_op.SUM, group=0)
    dist.all_reduce(numberOfSamples, op=dist.reduce_op.SUM, group=0)
    dist.all_reduce(average_time, op=dist.reduce_op.SUM, group=0)
    return weighted_loss, numberOfSamples, average_time

def main(rank, wsize):
	batchSize = 16
	epochs = 2
	learningRate = 0.01
	momentum = 0.9
	numWorkers = 1

	net = Net(i1, o1, o2, o3)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr = learningRate, momentum = momentum)

	glovePath = "../Data/glove.6B/glove.6B.100d.txt"
	trainData = "../Data/SST/trees/train.txt"
        devData = "../Data/SST/trees/dev.txt"
	testData = "../Data/SST/trees/test.txt"

	# transformations = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

	trainLoader, bszTrain = partition_dataset(trainData, glovePath, batchSize)
        devLoader, bszDev = partition_dataset(devData, glovePath, batchSize)
	testLoader, bszTest = partition_dataset(testData, glovePath, batchSize)

	weighted_loss, numberOfSamples, average_time = run(rank, wsize, net, optimizer, criterion, epochs, trainLoader, bszTrain, devLoader, use_cuda)

	if rank == 0:
        	print('{}, {}'.format((weighted_loss/numberOfSamples)[0], (average_time/dist.get_world_size())[0]))
		print("Final Weighted Loss - ",(weighted_loss/numberOfSamples))




if __name__ == "__main__":
    
    #if len(sys.argv) != 2:
    #    print("ERROR")
    #    sys.exit(1)
    
    dist.init_process_group(backend="mpi")#, world_size=int(sys.argv[1]))
    rank = dist.get_rank()
    wsize = dist.get_world_size()

    main(rank, wsize)
