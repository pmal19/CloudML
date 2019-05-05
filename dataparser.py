import numpy as np
import json
import codecs
import re

import csv

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def load_nli_data(path, choose=lambda x: True):
    print "Loading", path
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            if not choose(loaded_example):
                continue
            example = {}
            example["label"] = loaded_example["gold_label"]
            example["sentence_1"] = re.sub(r'([^\s\w]|_)+', '', loaded_example["sentence1"]).lower()
            example["sentence_2"] = re.sub(r'([^\s\w]|_)+', '', loaded_example["sentence2"]).lower()
            examples.append(example)
    return examples

def load_sst_data(
        path, top_node_only=True):
    dataset = convert_unary_binary_bracketed_data(path, top_node_only=top_node_only)
    return dataset

def load_quora_data(path):
    examples=[]
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(reader):
            if i < 1:
                continue
            if len(row) == 0:
                continue
            example = {}
            if len(row[3]) < 1 or len(row[4]) < 1:
                continue
            example['sentence_1'] = re.sub(r'([^\s\w]|_)+', '', row[3].lower()).lower()
            example['sentence_2'] = re.sub(r'([^\s\w]|_)+', '', row[4].lower()).lower()
            example['label'] = int(row[5])
            examples.append(example)
    return examples

# def convert_unary_binary_bracketed_data(
#         filename,
#         keep_fn=lambda x: True,
#         convert_fn=lambda x: x):
#     examples = []
#     with open(filename, 'r') as f:
#         for line in f:
#             example = {}
#             line = line.strip()
#             if len(line) == 0:
#                 continue
#             label = line[1]
#             if not keep_fn(label):
#                 continue
# 	        label = convert_fn(label)
#             example["label"] = label
#             example["sentence"] = line
#             example["tokens"] = []
#             example["transitions"] = []
#             words = example["sentence"].replace(')', ' )')
#             words = words.split(' ')

#             for index, word in enumerate(words):
#                 if word[0] != "(":
#                     if word == ")":
#                         # Ignore unary merges
#                         if words[index - 1] == ")":
#                             example["transitions"].append(1)
#                     else:
#                         # Downcase all words to match GloVe.
#                         example["tokens"].append(word.lower())
#                         example["transitions"].append(0)
#             example["example_id"] = str(len(examples))
#             example["sentence_1"] = re.sub(r'([^\s\w]|_)+', ''," ".join(example["tokens"])).lower()
#             for k in ['example_id','sentence','tokens', 'transitions']:
#             	del(example[k])
#             examples.append(example)
#     return examples

def span_to_example(words,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        id=''):
    label = words[0][1]
    if not keep_fn(label):
        return None
    label = convert_fn(label)

    example = {}
    example["label"] = label
    example["sentence_1"] = " ".join(words)
    example["tokens"] = []
    example["transitions"] = []
    for index, word in enumerate(words):
        if word[0] != "(":
            if word == ")":
                # Ignore unary merges
                if words[index - 1] == ")":
                    example["transitions"].append(1)
            else:
                # Downcase all words to match GloVe.
                example["tokens"].append(word)
                example["transitions"].append(0)
    example["example_id"] = id
    example["sentence_1"]=re.sub(r'([^\s\w]|_)+', ''," ".join(example["tokens"])).lower()
    return example

def convert_unary_binary_bracketed_data(
        filename,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x,
        top_node_only=False):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    if top_node_only:
        print("SST eval mode: Preserving only top node label.")
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            stack = []
            words = line.replace(')', ' )')
            words = words.split(' ')
            if top_node_only:
                example = span_to_example(words, keep_fn, convert_fn, str(len(examples)))
                if example is not None:
                    examples.append(example)
            else:
                for index, word in enumerate(words):
                    if word[0] != "(":
                        if word == ")":
                            start = stack.pop()
                            example = span_to_example(words[start:index + 1], keep_fn, convert_fn, str(len(examples)))
                            if example is not None:
                                examples.append(example)
                    else:
                        stack.append(index)
    return examples