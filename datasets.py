import time
import sys
import os
import re
import torch
# from torch import np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from itertools import accumulate

# sys.path.insert(0, '../../../')
# sys.path.insert(0, '../../')
# sys.path.insert(0, '../')

# from Models.blocks import *
from dataparser import *
from batcher import *
from readEmbeddings import *
# import Taskselector as task_selector

import pdb

class qoraDataset(Dataset):
    def __init__(self, nliPath, glovePath, transform = None):
    
        self.data = dataparser.load_quora_data(nliPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):

        s1 = self.pad(self.data[index]['sentence_1'].split())
        s2 = self.pad(self.data[index]['sentence_2'].split())

        s1 = self.embed(s1)
        s2 = self.embed(s2)
        
        label = self.data[index]['label']
        return (s1, s2), label

    def __len__(self):
        return len(self.data)

    def maxlength(self, data):
        maxSentenceLength = max([max(len(d['sentence_1'].split()),len(d['sentence_2'].split())) for d in data])
        return maxSentenceLength

    def pad(self, sentence):
        return sentence + (self.maxSentenceLength-len(sentence))*self.paddingElement

    def embed(self, sentence):
        vector = []
        for word in sentence:
            if str(word) in self.vocab:
                vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
            else:
                vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
        return vector


class sstDataset(Dataset):
    def __init__(self, sstPath, glovePath, transform = None, training = True):
        self.data = dataparser.load_sst_data(sstPath, (not training))
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
        return maxSentenceLength

    def pad(self, sentence):
        return sentence + (self.maxSentenceLength-len(sentence))*self.paddingElement

    def embed(self, sentence):
        vector = []
        for word in sentence:
            if str(word) in self.vocab:
                vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
            else:
                vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
        return vector


class nliDataset(Dataset):
    def __init__(self, nliPath, glovePath, transform = None):
    
        self.data = dataparser.load_nli_data(nliPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):

        s1 = self.pad(self.data[index]['sentence_1'].split())
        s2 = self.pad(self.data[index]['sentence_2'].split())

        s1 = self.embed(s1)
        s2 = self.embed(s2)
        
        label = LABEL_MAP[self.data[index]['label']]
        return (s1, s2), label

    def __len__(self):
        return len(self.data)

    def maxlength(self, data):
        maxSentenceLength = max([max(len(d['sentence_1'].split()),len(d['sentence_2'].split())) for d in data])
        return maxSentenceLength

    def pad(self, sentence):
        return sentence + (self.maxSentenceLength-len(sentence))*self.paddingElement

    def embed(self, sentence):
        vector = []
        for word in sentence:
            if str(word) in self.vocab:
                vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
            else:
                vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
        return vector