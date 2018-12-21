import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.util import mark_negation
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import nltk
import re


train_file = "audio_train.json"
test_file = "audio_dev.json"
train_data = []
print("Reading Training file")
with open(train_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        train_data.append(json_data)
test = {"data":[], "target":[], "target_names":["negative", "neutral", "positive"]}
with open(test_file, 'r') as json_file:
    for line in json_file.readlines():
        json_data = json.loads(line)
        train_data.append(json_data)

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
                if i%100000 == 0:
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

train = cleandata(train_data)
print("Preprocessing done")
print(len(train["data"]))
print("Starting training")

print("Starting training")
pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer = LemmaTokenizer(), stop_words='english', ngram_range=(1,2), max_df = 0.85)),
        ('clf', LinearSVC(verbose = 5, class_weight="balanced")),
    ])

pipeline.fit(train["data"], train["target"])
print("Trained")

test = cleandata(test_data)
y_predicted = pipeline.predict(y_vectors)
print(metrics.classification_report(test["target"], y_predicted,
                                    target_names=train["target_names"]))
cm = metrics.confusion_matrix(test["target"], y_predicted)
print(cm)
print(metrics.f1_score(test["target"], y_predicted, average='macro'))
import pickle
pickle.dump( pipeline, open( "pipeline.p", "wb" ) )
