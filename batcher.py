import numpy as np
import dataparser


def pad_with_maxlen(arr):
	padding_element=['<s>']
	maxlen=max([len(k) for k in arr])
	arr=[k+(maxlen-len(k))*padding_element for k in arr]
	arr=np.array(arr)
	arr=np.reshape(arr, (-1, maxlen))
	return arr

def maxLength(arr):
	return max([len(a) for a in arr])

def get_data(file_path, data_type="nli"):
	vocab_map = []
	if data_type == "nli":
		data = dataparser.load_nli_data(file_path)
		sentence_1 = [map(str.lower,str(x["sentence_1"])[:-2].split()) for x in data]
		sentence_2 = [map(str.lower,str(x["sentence_2"])[:-2].split()) for x in data]	

		return (pad_with_maxlen(sentence_1), pad_with_maxlen(sentence_2), data['label'])