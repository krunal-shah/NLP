import nltk
import sklearn_crfsuite
import eli5
from string import punctuation
import random
import numpy as np
import pickle

train_file_path = "../ner.txt"

print("Reading Training file")
train_sentences = []
train_sentences_tags = []
train_data = []
with open(train_file_path, 'r') as train_file:
    sentence = ""
    sentence_tags = []
    for line in train_file.readlines():
        line = line.rstrip("\n\r")
        words = line.split()
        if(len(words) == 2):
            sentence = sentence + words[0] + " "
            sentence_tags.append(words[1])
        else:
            if(sentence == ""):
                continue
            train_sentences.append(sentence)
            train_sentences_tags.append(sentence_tags)
            sentence = ""
            sentence_tags = []

train_data_sklearn = []
split = int(len(train_sentences)/10)
test_sentences = train_sentences[0:split]
train_sentences= train_sentences[split:]
test_sentences_tags = train_sentences_tags[0:split]
train_sentences_tags = train_sentences_tags[split:]

for sentence,gold_labels in zip(train_sentences, train_sentences_tags):
    tags = nltk.pos_tag(sentence.split())
    ret = []
    for tag,gold_label in zip(tags,gold_labels):
        tag = tag + (gold_label,)
        ret.append(tag)
    train_data_sklearn.append(ret)
test_data_sklearn = []
for sentence,gold_labels in zip(test_sentences, test_sentences_tags):
    tags = nltk.pos_tag(sentence.split())
    ret = []
    for tag,gold_label in zip(tags,gold_labels):
        tag = tag + (gold_label,)
        ret.append(tag)
    test_data_sklearn.append(ret)
    
def has_hyphen(word):
    if '-' in word:
        return True
    else:
        return False
    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:3]': postag[:3],
        'has_hyphen': has_hyphen(word),
        'len': (len(word) < 5)
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list

X_train = [sent2features(s) for s in train_data_sklearn]
y_train = [sent2labels(s) for s in train_data_sklearn]

X_test = [sent2features(s) for s in test_data_sklearn]
y_test = [sent2labels(s) for s in test_data_sklearn]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations = 100,
    all_possible_transitions=True,
)

crf.fit(X_train+X_test, y_train+y_test)

with open('crfsuite_model1.pkl','wb') as model_pkl:
    pickle.dump(crf, model_pkl)