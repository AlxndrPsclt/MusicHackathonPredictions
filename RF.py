# !/usr/bin/env python
#  -*- coding: utf-8 -*-

import sys
import csv
import pprint
import logging
import numpy as np
import pandas as pd
import itertools
#from matplotlib.ticker import LinearLocator, FormatStrFormatter


#from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

orig_stdout = sys.stdout
# f = file('./out.txt', 'w')
# sys.stdout = f

#fig = plt.figure()
#ax = fig.gca(projection='3d')


pp = pprint.PrettyPrinter(indent=2)
logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)


file_in_users=open("../users_res.csv", "r")
reader_users = csv.reader(file_in_users)

file_in_train=open("../train_res.csv", "r")
reader_train = csv.reader(file_in_train)

file_in_words=open("../words_res.csv", "r")
reader_words = csv.reader(file_in_words)

categories_words = reader_words.next()   # Le premier next permet de sauter la ligne contenant les noms des catégories
categories_users = reader_users.next()
categories_train = reader_train.next()

# Lecture des fichiers words train et users, convertissant chaque champ en float et stockant
# le tout dans un objet DataFrame qui permet de faire les joins
words = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_words], columns=categories_words)
users = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_users], columns=categories_users)
train = pd.DataFrame([ map(lambda x: float(x), row) for row in reader_train], columns=categories_train)

joined1 = pd.merge(train, words, on=["Artist", "User"])
attributes = pd.merge(joined1, users, on="User")

del train   # Suppression des trucs inutiles pour libérer un peu de mémoire
del words
del users
del joined1


attributes = attributes.apply(lambda x: MinMaxScaler().fit_transform(x))
pp.pprint(attributes[:10])


ratings=attributes.Rating.as_matrix()    # Conversion vers un numpy array avec la fonction as_matrix pour sklearn
# NB: les objets DataFrame permettent d'accéder très facilement aux colones attributes.Rating==attributes['Rating']
# On peut effectuer plein d'opérations dessus; voir la doc de Panda

del attributes["Artist"]       # Suppression des colonnes Artist, ratings des attributs, inutiles
del attributes["Rating"]

# for i in attributes.columns:
#     print i
attributes = attributes.as_matrix()

# test_attributes = attributes[:1000]
# test_ratings = ratings[:1000]
#
# train_attributes = attributes[1000:]
# train_ratings = ratings[1000:]

train_attributes, test_attributes, train_ratings, test_ratings = cross_validation.train_test_split\
    (attributes,ratings, test_size=0.2)

print("# # # # # # # # # # # # # # # # # # # # # # # ")


def rmse(ground_truth, predictions):
    weight = 100  # Pour lever la normalisation
    return (mean_squared_error(ground_truth * weight,predictions * weight))**0.5

RMSE = make_scorer(rmse, greater_is_better=False)

n_estimators_params = range(5,65)
max_feature_params = ['sqrt', 'log2', .5,]


for max_feature in max_feature_params:
    print "******************************************************************"
    print("Using " + str(max_feature))
    best_scores=list()
    for n_estimator in n_estimators_params:
        print("______________________________________________________________________________")
        print("Calculating for max_features = " + str(max_feature))
        rf = RandomForestRegressor(n_estimators=n_estimator, max_features=max_feature, n_jobs=-1, verbose=0)  # initialisation
        rf.fit(train_attributes, train_ratings)
        scores = cross_validation.cross_val_score(rf, attributes, ratings, scoring='mean_squared_error', cv=10)

        scores_a = np.array(abs(np.array(scores))*10000)**.5
        print("    RMSE array with " + str(n_estimator) + " trees is RMSE = " + str(scores_a))
        print ("    Mean RMSE with " + str(n_estimator) + " trees is " + str(scores_a.mean()))
        print ("    Min RMSE with " + str(n_estimator) + " trees is " + str(scores_a.min()))
        best_scores.append(scores_a.min())





# xy = a = [(8, 4), (10, 4), (12, 4), (12, 6), (14, 4), (14, 6), (16, 4), (16, 6), (16, 8), (18, 4), (18, 6), (18, 8), (20, 4), (20, 6), (20, 8), (20, 10), (22, 4), (22, 6), (22, 8), (22, 10), (24, 4), (24, 6), (24, 8), (24, 10), (24, 12), (26, 4), (26, 6), (26, 8), (26, 10), (26, 12), (28, 4), (28, 6), (28, 8), (28, 10), (28, 12), (28, 14), (30, 4), (30, 6), (30, 8), (30, 10), (30, 12), (30, 14)]
# x, y = np.array([ a for (a,_) in xy ]), np.array([b for (_,b) in xy])
#
# z = np.array(confusion_matrixes)
#
# surf = ax.scatter(x, y, z)
# ax.set_zlim(z.min(), z.max())
#
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# #fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()

# rf = RandomForestRegressor(max_features="sqrt", n_jobs=-1, verbose=1, warm_start=True)  # initialisation
# rf.fit(train_attributes,train_ratings)

# scores = cross_validation.cross_val_score(rf, attributes, ratings, cv=10, scoring=RMSE)
# print("cross validation score result : " + str(scores))

# print("Predicted values :")
# result_ratings = rf.predict(test_attributes)
# pp.pprint(result_ratings[:20])
#
# print("Differences :")
# diff=result_ratings - test_ratings
# pp.pprint(diff[:20])
#
# rmse = mean_squared_error(100 * test_ratings,100*result_ratings)**0.5
# print("RMSE = "+str(rmse))



