#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import re
import itertools as it
import pprint
pp = pprint.PrettyPrinter(indent=2)

import tempfile

from sknn.platform import cpu64, threading8
from sknn.mlp import Regressor, Layer
import sys

import logging
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn import cross_validation

from sklearn.metrics import mean_squared_error as MSE

from joblib import Parallel, delayed
from joblib import load, dump



import numpy as np
import pandas as pd


NN_ITERATIONS=10
CV_ITERATIONS=10

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
prod = list(it.product(l1,l2))

usefullCombinations = list(it.compress(prod, satisfy))

#usefullCombinations=[(12, 6), (16, 4), (16, 6), (22, 6), (22, 10), (24, 6), (24, 12), (28, 6)]

usefullCombinations=[(40, 15), (10, 5), (45,9)]

learningRates = np.array(range(5,25,5)+range(25,100,14))/1000.

nnCycles = range(10,30,2)

def my_callback(event, **variables):
    print(event)        # The name of the event, as shown in the list above.
    print(variables)    # Full dictionary of local variables from training loop

def trainAndTest(l1,l2,i,bestRMSEOutput, meanRMSEOutput):
    nn = Regressor(
        layers=[
            Layer("Rectifier", units=l1),
            Layer("Tanh", units=l2),
            Layer("Linear")],
        learning_rate=0.02,
        n_iter=NN_ITERATIONS)

    #CrossvalidationMode
    scores = cross_validation.cross_val_score(nn, attributes, ratings, scoring='mean_squared_error', cv=CV_ITERATIONS)

    print("Scores for "+str(l1)+" "+str(l2)+" :")
    scores_a=(abs(np.array(scores))*10000)
    scores_a=scores_a**.5

    bestRMSEOutput[i]=scores_a.min()
    meanRMSEOutput[i]=scores_a.mean()


folder = tempfile.mkdtemp()
bestRMSE_name = os.path.join(folder, 'bestRMSE')
bestRMSE = np.memmap(bestRMSE_name, dtype=np.dtype(float), shape=(len(usefullCombinations),), mode='w+')
meanRMSE_name = os.path.join(folder, 'meanRMSE')
meanRMSE = np.memmap(meanRMSE_name, dtype=np.dtype(float), shape=(len(usefullCombinations),), mode='w+')
duringTrainingRMSE_names = [os.path.join(folder, 'duringTrainingRMSE'+str(i)) for i in range(8)]
duringTrainingRMSE = [np.memmap(meanRMSE_name[i], dtype=np.dtype(float), shape=(NN_ITERATIONS), mode='w+')]

trainAndTest(45,10,0, bestRMSE, meanRMSE)
