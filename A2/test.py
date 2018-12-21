import nltk
import sklearn_crfsuite
import eli5
import random
import numpy as np
import sys
import pickle

input_file = sys.argv[1]
output_file = sys.argv[2]
model_path = sys.argv[3]
# with_file = sys.argv[4]

test_sentences = []
test_sentences_tags = []
with open(input_file, 'r', encoding='ISO-8859-2') as train_file:
    sentence = ""
    sentence_tags = []
    for line in train_file.readlines():
        line = line.rstrip("\n\r")
        words = line.split()
        if(len(words) == 2):
            sentence = sentence + words[0] + " "
            sentence_tags.append(words[1])
        elif(len(words) == 1):
            sentence = sentence + words[0] + " "
            sentence_tags.append("O")
        else:
            if(sentence == ""):
                continue
            test_sentences.append(sentence)
            test_sentences_tags.append(sentence_tags)
            sentence = ""
            sentence_tags = []

data_sklearn = []
for sentence,gold_labels in zip(test_sentences, test_sentences_tags):
    tags = nltk.pos_tag(sentence.split())
    ret = []
    for tag,gold_label in zip(tags,gold_labels):
        tag = tag + (gold_label,)
        ret.append(tag)
    data_sklearn.append(ret)

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


X_test = [sent2features(s) for s in data_sklearn]
y_test = [sent2labels(s) for s in data_sklearn]

#Load model
model_pkl = open(model_path, 'rb')
crf = pickle.load(model_pkl)
predicted = crf.predict(X_test)

counter = 0
for my, predicted_sentence in zip(y_test, predicted):
    for my_word, predicted_word in zip(my, predicted_sentence):
        if my_word != predicted_word:
            counter += 1
print(counter)

with open(output_file, 'w', encoding='ISO-8859-2') as write_file:
    for sentence, predicted_sentence in zip(test_sentences, predicted):
        for word, predicted_word in zip(sentence.split(), predicted_sentence):
            write_file.write(word + " " + predicted_word + "\n")
        write_file.write("\n")
    write_file.write("\n")
