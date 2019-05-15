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
		random.seed(seed)
		data_len = len(data)
		indexes = [x for x in range(0, data_len)]
		random.shuffle(indexes)

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
	data_set = DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=1)
	return data_set, bsz


def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size
  


def send_gradients_to_server_and_update_model(model):
	# Sending gradients to server
	tag = 0
	dist.send(tensor=torch.Tensor([tag]), dst=0)
	for param in model.parameters():
		dist.send(tensor=param.grad.data, dst=0)
	# Recieving data from server
	for param in model.parameters():
		dist.recv(tensor=param.data, src=0)


def get_updated_model(model, workers_handle):
	# rank_worker_finished = torch.zeros(dist.get_world_size() - 1)
	# rank_worker_finished[rank - 1] = 1
	tag = 1
	dist.send(tensor=torch.Tensor([tag]), dst=0)
	# dist.send(tensor=rank_worker_finished, dst=0)
	# print('Rank {} epoch end waiting on params barrier'.format(dist.get_rank()))
	# dist.barrier(workers_handle)
	for name, param in model.named_parameters():
		# print('Recv Rank ',dist.get_rank(),' name - ',name)
		dist.recv(tensor=param.data, src=0)


def send_updated_model_epoch_end(model):
	for dst in range(1, dist.get_world_size()):
		# print('Rank {} Sending epoch end'.format(dst))
		for name, param in model.named_parameters():
			# print('Send Rank ',dist.get_rank(),' name - ',name)
			dist.send(tensor=param.data, dst=dst)


def send_updated_model(model, dst):
	for param in model.parameters():
		dist.send(tensor=param.data, dst=dst)



def runServer(model, optimizer, epochs):
	model.zero_grad()
	for param in model.parameters(): #???
		param.sum().backward()
	workers = list(range(1, dist.get_world_size()))
	workers_handle = dist.new_group(workers)
	tag = 0
	tag = torch.Tensor([tag])
	for epoch in range(epochs):
		# print('server epoch start ',epoch)
		workers_finished = [False]*(dist.get_world_size() - 1)
		while not reduce((lambda x, y: x and y), workers_finished):
			src = dist.recv(tensor=tag)
			if tag == 0:
				model.zero_grad()
				optimizer.zero_grad()
				for param in model.parameters():
					dist.recv(tensor=param.grad.data, src=src)
				optimizer.step()
				send_updated_model(model, src)
			elif tag == 1:
				workers_finished[src - 1] = True
			# print(workers_finished, reduce((lambda x, y: x and y),workers_finished))
		# print(workers_finished, workers_finished.sum(), dist.get_world_size() - 1)
		# print(workers_finished, dist.get_world_size() - 1)
		for dst in workers:
			send_updated_model(model, dst)
		# send_updated_model_epoch_end(model)
		# dist.barrier(workers_handle)
		# print('server epoch end ',epoch)


def runWorker(rank, size, model, optimizer, criterion, epochs, trainLoader, bsz, devLoader, use_cuda, batchSize, devbatchSize, inp_dim):
	workers = list(range(1, dist.get_world_size()))
	workers_handle = dist.new_group(workers)
	torch.manual_seed(1234)
	num_batches = math.ceil(len(trainLoader.dataset) / float(bsz))
	numberOfSamples = 0
	start_time = time.monotonic()
	for epoch in range(epochs):
		epoch_loss = 0.0
		numberOfSamples = 0
		# print('worker rank ',rank,' epoch start ',epoch)
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

			send_gradients_to_server_and_update_model(model)

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
				dev_acc = (100. * n_correct.data)/n_total

				print('Rank {}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDev Loss: {:.6f}\tDev Acc: {:.6f}'.format(rank, epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.data, dev_loss.data, dev_acc))
			
			print('Rank {}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(rank, epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.item()))
		print('Rank {}, epoch {} avg loss : {}'.format(dist.get_rank(), epoch, epoch_loss/num_batches))
		#     print('Rank - {} - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(rank, epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))
		# print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
		# dist.barrier(workers)
		# get_updated_model(model)
		get_updated_model(model, workers_handle)
		dist.barrier(workers_handle)
		# print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
	end_time = time.monotonic()
	average_time = torch.Tensor([(end_time - start_time)/epochs])
	weighted_loss = torch.Tensor([(epoch_loss/num_batches) * numberOfSamples])
	numberOfSamples = torch.Tensor([numberOfSamples])
	dist.all_reduce(weighted_loss, op=dist.reduce_op.SUM, group=workers_handle)
	dist.all_reduce(numberOfSamples, op=dist.reduce_op.SUM, group=workers_handle)
	dist.all_reduce(average_time, op=dist.reduce_op.SUM, group=workers_handle)
	return weighted_loss, numberOfSamples, average_time


def main(rank, wsize):
	batchSize = 16
	epochs = 2
	learningRate = 0.01
	momentum = 0.9
	numWorkers = 1
	use_cuda = torch.cuda.is_available()

	model = BiLSTMSentiment(100, 100, 100, 5, use_cuda, batchSize) 
	criterion = nn.CrossEntropyLoss()
	if(use_cuda):
		model.cuda()
	if(use_cuda):
		criterion = nn.CrossEntropyLoss().cuda()
	else:
		criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)

	glovePath = "../Data/glove.6B/glove.6B.100d.txt"
	trainData = "../Data/SST/trees/train.txt"
	devData = "../Data/SST/trees/dev.txt"
	testData = "../Data/SST/trees/test.txt"

	# transformations = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

	trainLoader, bszTrain = partition_dataset(trainData, glovePath, batchSize)
	devLoader, bszDev = partition_dataset(devData, glovePath, batchSize)
	# testLoader, bszTest = partition_dataset(testData, glovePath, batchSize)
	print('Rank {} - Data loaded of len {}'.format(rank, len(trainLoader)))

	if rank == 0:
		runServer(model, optimizer, epochs)
	else:
		weighted_loss, numberOfSamples, average_time = runWorker(rank, wsize, model, optimizer, criterion, epochs, trainLoader, bszTrain, devLoader, use_cuda, batchSize, batchSize, 100)
		if rank == 1:
			print('Rank {} Loss - {}, Avg Time - {}'.format(rank, (weighted_loss/numberOfSamples)[0], (average_time/dist.get_world_size() - 1)[0]))



def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	os.environ['WORLD_SIZE'] = world_size
	# os.environ['RANK'] = rank // perform OS call hostname and rank to check
	# initialize the process group
	dist.init_process_group(backend='gloo', rank=rank, world_size=world_size) # or 'nccl'
	# Explicitly setting seed to make sure that models created in two processes
	# start from same random weights and biases.
	torch.manual_seed(42)
	
	
def cleanup():
	dist.destroy_process_group()


def setupAndCall(rank, world_size):
	setup(rank, world_size)
	print("MP Rank - {}".format(rank))
	hostname = socket.gethostname()
	# runDistCollectives(rank, world_size, hostname)
	main(rank, world_size)
	cleanup()
	
	
def runDistCollectives(rank, world_size, hostname):
	print("I am {} of {} in {}".format(rank, world_size, hostname))
	tensor = torch.zeros(1)
	if rank == 0:
		tensor += 1
		# Send the tensor to process 1
		dist.send(tensor=tensor, dst=1)
	else:
		# Receive tensor from process 0
		dist.recv(tensor=tensor, src=0)
	print('Rank {} has data {}'.format(rank, tensor[0]))
	
	
def spawnProcesses(fn, world_size):
	mp.spawn(fn,
			 args=(world_size,),
			 nprocs=world_size,
			 join=True)
	

if __name__ == "__main__":
	world_size = sys.argv[1]
	print("World Size : {}", world_size)
	spawnProcesses(setupAndCall, world_size)
