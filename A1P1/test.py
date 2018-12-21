import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import json
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = UserWarning)
	from nltk.sentiment.util import mark_negation
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
import pickle

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def cleandata(train_data):
    i = 0
    train = {"data":[], "target":[], "target_names":["negative", "neutral", "positive"]}
    for data in train_data:
        i = i + 1
        if i%10000 == 0:
            print(i)
        rating = int(data["overall"])
        text = data["reviewText"]
        summary = data["summary"]
        text = summary + ". " + summary + ". " + summary + ". " + text
        clean = ""
        text = re.sub('\s\''," ",text)
        text = re.sub('\'((\s)|($))', " ", text)
        text = re.sub('f[u*%$#@%][c*%$#@%][k*%$#@%]'," fuck", text)
        text = " ".join(mark_negation(nltk.word_tokenize(text), double_neg_flip=True, shallow=True))
        for ch in ["&quot", "\"", "\\\""]:
            if ch in text:
                text = text.replace(ch, " ")
        train["data"].append(text)
        category = 0
        if rating <= 2:
            category = 0
        elif rating == 3:
            category = 1
        else:
            category = 2
        train["target"].append(category)
    return train


input_file = sys.argv[1]
output_file = sys.argv[2]
output = open(output_file,'w') 

test_file = input_file
test_data = []
with open(test_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        test_data.append(json_data)

pipeline = pickle.load( open( "pipeline.p", "rb" ))
test = cleandata(test_data)
y_predicted = pipeline.predict(test["data"])
y_new_predicted = [((2*x)+1) for x in y_predicted]
for item in y_new_predicted:
    output.write(str(item)+"\n")
output.close()
