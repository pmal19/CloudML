import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.autograd import Variable


PAD_IDX=0
UNK_IDX=1
MAX_SENTENCE_LENGTH=200

def build_vocab(all_tokens,max_vocab_size=10000):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def build_vocab_pretrained(pretrained_path):
    words_to_load=100000
    with open(pretrained_path) as f:
        loaded_embeddings_ft = np.zeros((words_to_load+1, 300))
        token2id = {}
        id2token = ['<pad>']
        ordered_words_ft = []
        for i, line in enumerate(f):
            if i >= words_to_load:
                break
            s = line.split()
            loaded_embeddings_ft[i+1, :] = np.asarray(s[1:])
            token2id[s[0]] = i
            id2token.append(s[0])
            ordered_words_ft.append(s[0])
        loaded_embeddings_ft[PAD_IDX,:] = np.zeros(300)
        token2id['<pad>'] = PAD_IDX
    return token2id, id2token, loaded_embeddings_ft

def token2index_dataset(tokens_data,token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data
class YelpDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self,  data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        #print(len(self.data_list),len(self.target_list))
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def yelp_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])),
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    #print(label_list[:10])
    return [torch.from_numpy(np.array(data_list)).long(), torch.LongTensor(length_list), torch.LongTensor(label_list)]

def yelp_collate_func_rnn(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        if datum[0]!=[]:
            label_list.append(datum[2])
            length_list.append(datum[1])
    # padding
    for datum in batch:
        if datum[0]!=[]:
            padded_vec = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])),
                                mode="constant", constant_values=0)
            data_list.append(padded_vec)
    #print(label_list[:10])
    _, idx_sort = torch.sort(torch.tensor(length_list), dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)

    length_list = list(torch.tensor(length_list)[idx_sort])
    idx_sort = Variable(idx_sort)
    data_list = torch.tensor(data_list).index_select(0,idx_sort)
    return [torch.from_numpy(np.array(data_list)).long(), torch.LongTensor(length_list), idx_unsort, torch.LongTensor(label_list)]
