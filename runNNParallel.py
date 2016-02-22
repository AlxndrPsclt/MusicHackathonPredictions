#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import re
import itertools as it
import pprint
pp = pprint.PrettyPrinter(indent=2)

import tempfile

from sknn.platform import cpu64, threading
from sknn.mlp import Regressor, Layer
import sys

import logging
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn import cross_validation

from sklearn.metrics import mean_squared_error as MSE

from joblib import Parallel, delayed
from joblib import load, dump



import numpy as np
import pandas as pd


NN_ITERATIONS=100
CV_ITERATIONS=2

file_in_users=open("/home/pascault/data/users_res.csv","r")
reader_users = csv.reader(file_in_users)

file_in_train=open("/home/pascault/data/train_res.csv","r")
reader_train = csv.reader(file_in_train)

file_in_words=open("/home/pascault/data/words_res.csv","r")
reader_words = csv.reader(file_in_words)

categories_words = reader_words.next()
categories_users = reader_users.next()
categories_train = reader_train.next()



words = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_words], columns=categories_words)
users = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_users], columns=categories_users)
train = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_train], columns=categories_train)

joined1 = pd.merge(train, words, on=["Artist","User"])
del(train)
del(words)
attributes = pd.merge(joined1, users, on="User")
attributes=attributes.apply(lambda x: MinMaxScaler().fit_transform(x))
del(users)

del(joined1)

pp.pprint(attributes[:10])
pp.pprint(attributes.Rating[:10])

ratings=attributes.Rating.as_matrix()

del attributes["Artist"]
del attributes["Rating"]

attributes = attributes.as_matrix()


print("#######################")


#####Choosing the coeff to run the tests
l1=[135, 110, 85, 60, 40, 30, 20, 10]
l2=[60, 40, 30, 20, 10]

l1=[50, 20, 10]
l2=[20, 10]

l1=range(8,31,2)
l2=range(4,17,2)


prod = list(it.product(l1,l2))

satisfy = map(lambda (a,b): a>=2*b, prod)

usefullCombinations = list(it.compress(prod, satisfy))

usefullCombinations=[(12, 6), (16, 4), (16, 6), (22, 6), (22, 10), (24, 6), (24, 12), (28, 6)]



def trainAndTest(l1,l2,i,bestRMSEOutput, meanRMSEOutput):
    nn = Regressor(
        layers=[
            Layer("Sigmoid", units=l1),
            Layer("Sigmoid", units=l2),
            Layer("Linear")],
        learning_rate=0.02,
        n_iter=NN_ITERATIONS)

    #CrossvalidationMode
    #scores = cross_validation.cross_val_score(nn, attributes, ratings, scoring='mean_squared_error', cv=CV_ITERATIONS)

    #No Crossvalidation; run only once on random split data
    scores=[]
    attributes_train, attributes_test, ratings_train, ratings_test = cross_validation.train_test_split(attributes, ratings, test_size=0.10, random_state=42)

    print(len(attributes_train))
    print(len(attributes_test))
    print(len(ratings_train))
    print(len(ratings_test))

    nn.fit(attributes_train, ratings_train)
    ratings_result = nn.predict(attributes_test)

    mse = MSE(ratings_test, ratings_result)**.5
    scores.append(mse)

#    print("Scores for "+str(l1)+" "+str(l2)+" :")
    scores_a=(abs(np.array(scores))*10000)
    scores_a=scores_a**.5

    bestRMSEOutput[i]=scores_a.min()
    meanRMSEOutput[i]=scores_a.mean()


folder = tempfile.mkdtemp()
bestRMSE_name = os.path.join(folder, 'bestRMSE')
bestRMSE = np.memmap(bestRMSE_name, dtype=np.dtype(float), shape=(len(usefullCombinations),), mode='w+')
meanRMSE_name = os.path.join(folder, 'meanRMSE')
meanRMSE = np.memmap(meanRMSE_name, dtype=np.dtype(float), shape=(len(usefullCombinations),), mode='w+')

#trainAndTest(22,10,0, bestRMSE, meanRMSE)
Parallel(n_jobs=8)(delayed(trainAndTest)(l1,l2,i, bestRMSE, meanRMSE) for (i,(l1,l2)) in enumerate(usefullCombinations))

print("Tested configurations :")
print(usefullCombinations)

print("Best RMSE :")
print(bestRMSE)

print("Mean RMSE :")
print(meanRMSE)

print("Best confguration :")
print(bestRMSE.min())
print(usefullCombinations[bestRMSE.argmin()])

#nn = Regressor(
    #layers=[
        #Layer("Sigmoid", units=40),
        #Layer("Sigmoid", units=20),
        #Layer("Sigmoid", units=8),
        #Layer("Linear")],
    #learning_rate=0.02,
    #n_iter=NN_ITERATIONS)

#pipeline = Pipeline([
#        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
#        ('neural network', nn)])

#pipeline.fit(attributes_train, ratings_train)

#scores = cross_validation.cross_val_score(nn, attributes, ratings, scoring='mean_squared_error', cv=CV_ITERATIONS)

#print("Predicted values :")
#ratings_result = pipeline.predict(attributes_test)
#pp.pprint(ratings_result[:10])
#
#print("Expected values :")
#pp.pprint(ratings_test[:10])
#
##print("Differences :")
##diff=result_ratings - test_ratings
##pp.pprint(diff[:10])
##
##rmse = diff.mean()
#rmse = MSE(100*ratings_test, 100*ratings_result)**.5
#print("RMSE = "+str(rmse))

#print("Scores :")
#scores_a=(-np.array(scores)*10000)
#scores_a=scores_a**.5
#print(scores_a)
#print("Mean score :")
#print(np.array(scores_a).mean())
