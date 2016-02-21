#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
import itertools
import pprint
pp = pprint.PrettyPrinter(indent=2)
from sknn.platform import cpu64, threading
from sknn.mlp import Regressor, Layer
import numpy
import sys
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn import cross_validation

from sklearn.metrics import mean_squared_error as MSE

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

import numpy as np
import pandas as pd

NN_ITERATIONS=1
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

joined1 = pd.merge(train, words, on=["Artist","User"])
del(train)
del(words)
attributes = pd.merge(joined1, users, on="User")
attributes=attributes.apply(lambda x: MinMaxScaler().fit_transform(x))
del(users)

del(joined1)

pp.pprint(attributes[:10])

#words = { (row[0], row[1]) : list(itertools.compress(row,mask_words)) for row in reader_words }
#users = { row[0] : list(itertools.compress(row,mask_users)) for row in reader_users }
#train = { (row[0], row[2]) : list(itertools.compress(row,mask_train)) for row in reader_train }

pp.pprint(attributes.Rating[:10])
#attributes.Rating=attributes['Rating'].apply(lambda x: MinMaxScaler().fit_transform(x))
#attributes.Rating=MinMaxScaler().fit_transform(attributes.Rating)
#pp.pprint(attributes.Rating[:10])
ratings=attributes.Rating.as_matrix()

del attributes["Artist"]
del attributes["Rating"]

attributes = attributes.as_matrix()


#test_attributes = attributes[:5000]
#test_ratings = ratings[:5000]

print("#######################")

#attributes_train, attributes_test, ratings_train, ratings_test = cross_validation.train_test_split(attributes, ratings, test_size=0.10, random_state=42)

nn = Regressor(
    layers=[
        Layer("Sigmoid", units=40),
        Layer("Sigmoid", units=20),
        Layer("Sigmoid", units=8),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=NN_ITERATIONS)

#pipeline = Pipeline([
#        ('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),
#        ('neural network', nn)])

#pipeline.fit(attributes_train, ratings_train)

scores = cross_validation.cross_val_score(nn, attributes, ratings, scoring='mean_squared_error', cv=CV_ITERATIONS)

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

print("Scores :")
print(scores)
print("Mean score :")
print(np.array(scores).mean())
