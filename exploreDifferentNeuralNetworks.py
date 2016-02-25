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
CV_ITERATIONS=5

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
l1=[100, 85, 60, 40, 35, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10]
l2=[60, 40, 30, 20, 15, 10, 5]
l3=[20, 15, 10, 5]

nn = Regressor(
    layers=[
        Layer("Rectifier", units=10),
        Layer("Tanh", units=10),
        Layer("Tanh", units=10),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=NN_ITERATIONS)

attributes_train, attributes_test, ratings_train, ratings_test = cross_validation.train_test_split(attributes, ratings, test_size=0.10, random_state=42)

chunksL1=[l1[i:i+len(l1)/10] for i in xrange(0, len(l1), len(l1)/10)]


gs = GridSearchCV(nn, param_grid={
    'hidden0__units': l1,
    'hidden1__units': l2,
    'hidden2__units': l3})
gs.fit(attributes_train, ratings_train)


