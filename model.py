import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, hidden_size, emb_dim, use_pretrained = 'n', pretrained_vecs=None):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        #embed dimension should be atleast 100
        if use_pretrained=='y':
            pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float().to(device)
            self.embed = nn.Embedding.from_pretrained(pretrained_vecs_tensor).to(device)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0).to(device)
        self.linear = nn.Linear(emb_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,2)

    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        #data.type()
        #print(data.type())
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()

        # return logits
        out = F.relu(self.linear(out.float()))
        out=self.linear2(out)
        return out

    
class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, hidden_size_rnn, vocab_size,use_pretrained='n', pretrained_vecs=None, num_layers=1):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()
        if use_pretrained=='y':
            pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float().to(device)
            self.embed = nn.Embedding.from_pretrained(pretrained_vecs_tensor).to(device)
        self.num_layers, self.hidden_size, self.hidden_size_rnn = num_layers, hidden_size, hidden_size_rnn
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0).to(device)
        self.rnn = nn.GRU(emb_size, hidden_size_rnn, num_layers, batch_first=True, bidirectional=True) #First dimension is the batch size
        self.linear = nn.Linear(2*hidden_size_rnn, hidden_size)
        self.linear2=nn.Linear(hidden_size,2)

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size_rnn)

        return hidden

    def forward(self, x, lengths,unsort, data_parallel=False):
        # reset hidden state

        batch_size, seq_len = x.size()
        if not data_parallel:
            self.hidden = self.init_hidden(batch_size).to(device)
        else:
            self.hidden = self.module.init_hidden(batch_size).to(device)
        #print(x.type())
        # get embedding of characters
        embed = self.embedding(x)
        # pack padded sequence
        #pytorch wants sequences to be in decreasing order of lengths
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, self.hidden = self.rnn(embed, self.hidden)
        # undo packing
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # sum hidden activations of RNN across time
        rnn_out = torch.sum(rnn_out, dim=1)
        #print(rnn_out.shape)
        #print(self.hidden.shape)
        hidden=self.hidden.transpose(0,1).contiguous().view(batch_size, -1)
        #print(hidden.shape)
        hidden=hidden.index_select(0,unsort)
        hidden = F.relu(self.linear(hidden))
        out=self.linear2(hidden)
        return out

class CNN(nn.Module):
    """
    CNN classification model
    """
    def __init__(self, vocab_size, hidden_size,hidden_cnn, kernel_size, emb_dim,use_pretrained='n', pretrained_vecs=None):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(CNN, self).__init__()
        # pay attention to padding_idx 
        #embed dimension should be atleast 100
        if use_pretrained=='y':
            pretrained_vecs_tensor=torch.from_numpy(pretrained_vecs).float().to(device)
            self.embed = nn.Embedding.from_pretrained(pretrained_vecs_tensor).to(device)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0).to(device)
        self.conv1 = nn.Conv1d(emb_dim,hidden_cnn,kernel_size=kernel_size,padding=1)
        self.conv2 = nn.Conv1d(hidden_cnn,hidden_cnn,kernel_size=kernel_size,padding=1)
        self.linear = nn.Linear(hidden_cnn,hidden_size)
        self.linear2=nn.Linear(hidden_size,2)

    def forward(self, data, length):
        #data.type()
        #print(data.type())
        batch_size, seq_len = data.size()
        embed = self.embed(data)
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        # return logits
        hidden=torch.max(hidden,dim=1)[0]
        hidden = F.relu(self.linear(hidden))
        out=self.linear2(hidden)
        return out
