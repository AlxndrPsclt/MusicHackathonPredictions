#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
import itertools as it
import pprint
pp = pprint.PrettyPrinter(indent=2)
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


import numpy as np
import pandas as pd


NN_ITERATIONS=8
CV_ITERATIONS=6

file_in_users=open("/home/pascault/data/users_res.csv","r")
reader_users = csv.reader(file_in_users)

file_in_train=open("/home/pascault/data/train_res.csv","r")
reader_train = csv.reader(file_in_train)

file_in_words=open("/home/pascault/data/words_res.csv","r")
reader_words = csv.reader(file_in_words)

categories_words = reader_words.next()
categories_users = reader_users.next()
categories_train = reader_train.next()


def getMasks(categories, keywords):
    mask=map(lambda x: x in keywords, categories)
    antimask=map(lambda x: not x, mask)
    return (mask, antimask)

mask_words, antimask_words = getMasks(categories_words, ["Artist", "User"])
mask_users, antimask_users = getMasks(categories_users, ["User"])
mask_train, antimask_train = getMasks(categories_train, ["Artist", "User"])

words = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_words], columns=categories_words)
users = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_users], columns=categories_users)
train = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_train], columns=categories_train)

print("Words :")
pp.pprint(list(words.columns))
print("Users :")
pp.pprint(list(users.columns))
print("Train :")
pp.pprint(list(train.columns))

joined1 = pd.merge(train, words, on=["Artist","User"])
print("Joined1 :")
pp.pprint(joined1.columns)
del(train)
del(words)
attributes = pd.merge(joined1, users, on="User")
attributes=attributes.apply(lambda x: MinMaxScaler().fit_transform(x))
#attributes=attributes.apply(lambda x: MinMaxScaler().fit_transform(x))
print("Attributes :")
pp.pprint(list(attributes.columns))
del(users)

del(joined1)

pp.pprint(attributes[:10])
pp.pprint(attributes.Rating[:10])

ratings=attributes.Rating.as_matrix()

del attributes["Artist"]
del attributes["Rating"]

attributes = attributes.as_matrix()


print("#######################")
