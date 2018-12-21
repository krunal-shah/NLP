import nltk
import pickle
import json
from pprint import pprint
import nltk
from collections import Counter
import math
import numpy as np
from sklearn import metrics
import json
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import sys
import string

input_file = sys.argv[1]
output_file = sys.argv[2]
output = open(output_file,'w') 
print("Reading input file")
test_data = []
with open(input_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        test_data.append(json_data)
punctuations=string.punctuation
translator = str.maketrans(punctuations,' '*len(punctuations))

torch.manual_seed(1)

pickle_in = open("dict.pickle","rb")
dictionary = pickle.load(pickle_in)
pickle_in.close()
print(len(test_data))

stopwords_set = []
with open('./stopwords/english','r') as fp:
    line = fp.readline()
    while line:
        stopwords_set.append(line.rstrip())
        line = fp.readline()

# from nltk.corpus import stopwords
import re
import string


print(len(dictionary))



def clean(text):
    ret = ""
    for ch in punctuations:
        text = text.replace(ch, " ")
    for word in text.split():
        if word in stopwords_set:
            pass
        elif word in dictionary:
            ret = ret + word + " "
        else:
            ret = ret + "<UNK>" + " "
    return ret

def prepare_sequence(datas):
    ret = []
    ratings = []
    for data in datas:
        #print(data["overall"])
        arr = []
        rating = int(data["overall"])
        if rating <= 2:
            ratings.append(0)
        elif rating == 3:
            ratings.append(1)
        else:
            ratings.append(2)
        arr.append( dictionary["<START>"])
        i = 1
        text = clean(data["reviewText"])
        #print(text)
        for word in text.split():
            if i >= sentence_size-1:
                break
            if word in dictionary:
                arr.append(dictionary[word])
            else:
                arr.append(dictionary["<UNK>"])
            i = i + 1
        arr.append(dictionary["<EOS>"])
        i = i + 1
        #print(i)
        while i < sentence_size:
            #print(i)
            arr.append(dictionary["<PAD>"])
            i = i + 1
        ret.append(arr)
        if len(arr) != sentence_size:
            print("Problems with " + data["reviewText"])
    return ret, ratings


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        
        self.hidden2tag = nn.Linear(self.hidden_dim, 3)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = torch.load('model.pt')

batch_size = 256
embedding_dim = 512
hidden_dim = 256
sentence_size = 256
epoch_range = 2

import sklearn
from sklearn.metrics import f1_score

print("Testing")
test_data = test_data
batch_size = batch_size
model.batch_size = batch_size

prediction = []
test_target = []

num_itrs = int(len(test_data)/batch_size)
index = 0

for i in range(num_itrs):
    model.zero_grad()
    model.hidden = model.init_hidden()
    
    sentence_in, test_targets = prepare_sequence(test_data[index:index+model.batch_size])
    print(str(index) + " - " + str(index + model.batch_size))
    index = index + model.batch_size
    sentence_in = torch.LongTensor(sentence_in)
    test_targets = torch.LongTensor(test_targets)
    
    test_inputs, test_labels = Variable(sentence_in.cuda()), Variable(test_targets.cuda())
    
    test_scores = model(test_inputs.t())
    _, test_scores = torch.max(test_scores, 1)
    prediction = np.concatenate((prediction, test_scores.data.cpu().numpy()))
    test_target = np.concatenate((test_target, test_labels.data.cpu().numpy()))

model.batch_size = 1
batch_size = 1
while index < len(test_data):
    model.zero_grad()
    model.hidden = model.init_hidden()
    
    sentence_in, test_targets = prepare_sequence(test_data[index:index+model.batch_size])
    print(str(index))
    index = index + model.batch_size
    sentence_in = torch.LongTensor(sentence_in)
    test_targets = torch.LongTensor(test_targets)
    
    test_inputs, test_labels = Variable(sentence_in.cuda()), Variable(test_targets.cuda())
    
    test_scores = model(test_inputs.t())
    _, test_scores = torch.max(test_scores, 1)
    prediction = np.concatenate((prediction, test_scores.data.cpu().numpy()))
    test_target = np.concatenate((test_target, test_labels.data.cpu().numpy()))


# print(f1_score(test_target, prediction, average='macro'))

for predict in prediction:
    predict = int(predict)
    if predict <= 2:
        predict = 1
    elif predict == 3:
        predict = 3
    else:
        predict = 5
    output.write(str(predict)+'\n')
output.close()