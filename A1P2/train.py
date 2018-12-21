
# coding: utf-8

# In[ ]:


# TODO: Change this path while submitting
train_file = "./audio_train.json"

print("Reading Training file")
train_data = []
with open(train_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        train_data.append(json_data)

        test_file = "./audio_dev.json"
print("Reading test file")
test_data = []
with open(test_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        test_data.append(json_data)
import nltk


# In[ ]:


# stopwords_set = set(stopwords.words('english'))
punctuations=string.punctuation
translator = str.maketrans(punctuations,' '*len(punctuations))


# In[ ]:


dictionary = {}
count = {}
present = set()
index = 0
done = 0
for data in train_data:
    if done%10000 == 0:
        print(done)
    text = data["reviewText"] + " " + data["summary"] + " " + data["summary"]
    text = text.translate(translator)
    for word in text.split():
        if word in stopwords_set:
            pass
        else:
            if word in present:
                count[word] += 1
                if count[word] == 6:
                    dictionary[word] = index
                    #print(word)
                    index = index + 1
            else:
                count[word] = 0
            present.add(word)
    done += 1


# In[ ]:


dictionary["<START>"] = len(dictionary)
dictionary["<EOS>"] = len(dictionary)
dictionary["<UNK>"] = len(dictionary)
dictionary["<PAD>"] = len(dictionary)


# In[ ]:


import pickle
pickle_out = open("dict.pickle","wb")
pickle.dump(dictionary, pickle_out)
pickle_out.close()
pickle_out = open("train.pickle","wb")
pickle.dump(train_data, pickle_out)
pickle_out.close()
pickle_out = open("test.pickle","wb")
pickle.dump(test_data, pickle_out)
pickle_out.close()


# ## Train

# In[ ]:


import pickle
pickle_in = open("dict.pickle","rb")
dictionary = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("train.pickle","rb")
train_data = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open("test.pickle","rb")
test_data = pickle.load(pickle_in)
pickle_in.close()


# In[ ]:


print(len(train_data))
print(len(test_data))


# In[ ]:


import json
from pprint import pprint
import nltk
from collections import Counter
import math
import numpy as np
from sklearn import metrics
import json


# In[ ]:


stopwords_set = []
with open('./stopwords/english','r') as fp:
    line = fp.readline()
    while line:
        stopwords_set.append(line.rstrip())
        line = fp.readline()


# In[ ]:


# from nltk.corpus import stopwords
import re
import string


# In[ ]:


# stopwords_set = set(stopwords.words('english'))
punctuations=string.punctuation


# In[ ]:


print(len(dictionary))


# In[ ]:


batch_size = 256
embedding_dim = 512
hidden_dim = 256
sentence_size = 256
epoch_range = 2
iterations_per_epoch = int(len(train_data)/batch_size)


# In[ ]:


import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim, 3)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), self.batch_size, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out[-1])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[ ]:


model = LSTM(embedding_dim, hidden_dim, len(dictionary), batch_size).cuda()
loss_function = nn.NLLLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(training_data)
# tag_scores = model(inputs)
# print(tag_scores)


# In[ ]:


epoch_range = 2
model.batch_size = 1024
batch_size = 1024
batch_size = 1024
iterations_per_epoch = int(len(train_data)/batch_size)
for epoch in range(epoch_range):  # again, normally you would NOT do 300 epochs, it is toy data
    iteration = 0
    index = 0
    for iteration in range(iterations_per_epoch):
        model.zero_grad()

        model.hidden = model.init_hidden()

        sentence_in, targets = prepare_sequence(train_data[index:index+batch_size])
        index += batch_size
        sentence_in = torch.LongTensor(sentence_in)
        targets = torch.LongTensor(targets)
        
        test_inputs, test_labels = Variable(sentence_in.cuda()), Variable(targets.cuda())
        
        tag_scores = model(test_inputs.t())
        
        loss = loss_function(tag_scores, test_labels)
        loss.backward()
        
        print("epoch = %d iteration = %d loss = %f"%(epoch, iteration, loss))
        optimizer.step()


# In[ ]:


import pickle
pickle_out = open("model.pickle","wb")
pickle.dump(model, pickle_out)


# In[ ]:


torch.save(model, 'model_2_1024.pt')


# In[ ]:


import sklearn
from sklearn.metrics import f1_score

print("Testing")
test_data = test_data
batch_size = batch_size

prediction = []
test_target = []
model.batch_size = 1024
batch_size = 1024
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



print(f1_score(test_target, prediction, average='macro'))


# In[ ]:


print(len(prediction))


# In[ ]:


print(len(test_data))


# In[ ]:


print("Testing")
test_data = test_data
batch_size = batch_size
num_itrs = len(test_data)/batch_size
for i in range(num_itrs):
    model.zero_grad()
    model.hidden = model.init_hidden()
    
    sentence_in, targets = prepare_sequence(test_data[iteration:iteration+batch_size])
    sentence_in = torch.LongTensor(sentence_in)
    targets = torch.LongTensor(targets)

    test_inputs, test_labels = Variable(sentence_in.cuda()), Variable(targets.cuda())
    
    tag_scores = model(test_inputs.t())
    #print(tag_scores)
    #print(test_labels)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(tag_scores, test_labels)
    loss.backward()
    #print(loss)
    print("epoch = %d iteration = %d loss = %f"%(epoch, iteration, loss))
    optimizer.step()

