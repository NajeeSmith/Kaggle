# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:09:55 2018

@author: carto
"""

#Najee's Amazing Pet Adoption Prediction Guesser
#Importing Non-Classification Libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sb
import json 
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Importing and Splitting Data
#from sklearn.cross_validation import train_test_split
train_df = pd.read_csv('train.csv')

corpus = [] #Collection of the individual reviews
for i in range(0, 14993):
    review = re.sub('[^a-zA-Z]', ' ', str(train_df['Description'][i]))
    review = review.lower()
    review = review.split()
    stem = PorterStemmer()
    review = [stem.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '. join(review)
    corpus.append(review)
    
#Bag of Words Model via tokenization
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(max_features = 2000 )
X = vector.fit_transform(corpus).toarray()
y = train_df.iloc[:, 23].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Creating NB model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Prediction time!
guess = classifier.predict(X_test)

#Confusion Matrix to determine how many are correct
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, guess)

df_cm = pd.DataFrame(guess, range(2999),
                  range(1))
#plt.figure(figsize = (10,7))
sb.set(font_scale=1.4)#for label size
heatmap = sb.heatmap(df_cm, annot=False)# font size