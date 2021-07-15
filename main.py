
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import * 
from math import log, sqrt
import pandas as pd
import numpy as np


mails = pd.read_csv('spam.csv', encoding = 'latin-1')

#train test split

totalmails = mails['v2'].shape[0]
trainIndex, testIndex = list(), list()

for i in range(mails.shape[0]):
    if np.random.uniform(0,1) < 0.75:
        trainIndex +=[i]
    else:
        testIndex +=[i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]

trainData.reset_index(inplace = True)

# Visualization of data

spam_words = ' '.join(list(mails[mails['v1']=='spam']['v2']))
spam_wc = WordCloud(width =512,height=512).generate(spam_words)
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


def process_message(message,lower_case = True, stem = True, stop_words = True, gram =2):
    if lower_case:
        message= message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w =[]
        for i in range(len(words) - gram + 1):
            w.append([' '.join(words[i:i + gram])])
        return w
    if stop_words:
        sw = stopwords.word('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

"""
s_c = SpamClassifier(trainData, 'tf-idf')
s_c.train()
pred_s_c = s_c.predict(testData['v2'])
metrics(testData['label'],pred_s_c)
1.
"""