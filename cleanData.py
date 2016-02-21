#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re
from operator import truediv

file_in_users=open("../data/users.csv","r")
file_out_users=open("../data/users_tmp.csv","w")

reader_users = csv.reader(file_in_users)
writer_users = csv.writer(file_out_users)

file_in_train=open("../data/train.csv","r")
file_out_train=open("../data/train_res.csv","w")

reader_train = csv.reader(file_in_train)
writer_train = csv.writer(file_out_train)

file_in_words=open("../data/words.csv","r")
file_out_words=open("../data/words_tmp.csv","w")

reader_words = csv.reader(file_in_words)
writer_words = csv.writer(file_out_words)


listeCategories=[]

MISSING_VALUE = "-999"


def cleanHours(value):
    pattern= re.compile("(\d{,2})")
    m = re.search(pattern, value)
    if m.group(0)!="":
        value=m.group(0)
    elif "Less" in value:
        value=0
    elif value=="":
        value=MISSING_VALUE
    elif "More" in value:
        value=16
    else:
        print value
    return str(value)

def cleanHours2(value):
    print(value)

def cleanMusic(value):
    categories = ["Music has no particular interest for me",
    "I like music but it does not feature heavily in my life",
    "Music is no longer as important as it used to be to me",
    "Music is important to me but not necessarily more important",
    "Music is important to me but not necessarily more important than other hobbies or interests",
    "Music means a lot to me and is a passion of mine"]

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse=categories
    return reponse

def cleanMusicAsValue(value):
    categories = ["Music has no particular interest for me",
    "I like music but it does not feature heavily in my life",
    "Music is no longer as important as it used to be to me",
    "Music is important to me but not necessarily more important",
    "Music is important to me but not necessarily more important than other hobbies or interests",
    "Music means a lot to me and is a passion of mine"]

    reponse=[MISSING_VALUE]

    if value=="__NAME__":
        reponse=["MusicInterest"]
    elif value in categories:
        reponse = [str(categories.index(value))]
    else:
        if value!="":
            print value
    return reponse

def cleanRegion(value):
    if "Ireland" in value:
        value = "Northern Ireland"
    categories=['South', 'Midlands', 'North', 'Centre', 'Northern Ireland']

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse=categories
    return reponse

def cleanRegionAsValue(value):
    if "Ireland" in value:
        value = "Northern Ireland"
    categories=['South', 'Midlands', 'North', 'Centre', 'Northern Ireland']

    reponse=[MISSING_VALUE]

    if value=="__NAME__":
        reponse=["Region"]
    elif value in categories:
        reponse = [str(categories.index(value))]
    else:
        if value!="":
            print value
    return reponse

def cleanWork(value):
    categories=['Other', 'Full-time housewife / househusband',
                'Employed 30+ hours a week', 'Employed 8-29 hours per week',
                'Full-time student', 'Temporarily unemployed',
                'Retired from full-time employment (30+ hours per week)',
                'Self-employed', 'Retired from self-employment',
                'Prefer not to state', 'Part-time student',
                'In unpaid employment (e.g. voluntary work)',
                'Employed part-time less than 8 hours per week']

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse=categories
    return reponse

def cleanWorkAsValues(value):
    categories=['Other', 'Full-time housewife / househusband',
                'Employed 30+ hours a week', 'Employed 8-29 hours per week',
                'Full-time student', 'Temporarily unemployed',
                'Retired from full-time employment (30+ hours per week)',
                'Self-employed', 'Retired from self-employment',
                'Prefer not to state', 'Part-time student',
                'In unpaid employment (e.g. voluntary work)',
                'Employed part-time less than 8 hours per week']
    reponse=[MISSING_VALUE]

    if value=="__NAME__":
        reponse=[ "Employment" ]
    elif value in categories:
        reponse = [ str(categories.index(value)) ]
    else:
        if value!="":
            print value
    return reponse

def addMinusOneIfEmpty(value):
    if value=="":
        value="-1"
    return value

def addZeroIfEmpty(value):
    if value=="":
        value="0"
    return value

def addMissingValue(value):
    if value=="":
        value=MISSING_VALUE
    return value

def replaceMissingValueByMean(value, meanList, index):
    if value==MISSING_VALUE:
        value=meanList[index]
    return value

def cleanGender(value):
    if value=="Male":
        value=1
    elif value=="Female":
        value=0
    else:
        print value
    return str(value)

def cleanHeardOf(value):
    if "know" in value:
        value = "Dont know if I heard"
    categories = ["Heard of" ,
                  "Never heard of" ,
                  "Heard of and listened to music EVER" ,
                  "Ever heard of" ,
                  "Ever heard music by",
                  "Dont know if I heard",
                  "Listened to recently",
                  "Heard of and listened to music RECENTLY"]
    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse=categories
    return reponse

def cleanHeardOfAsValue(value):
    if "know" in value:
        value = "Dont know if I heard"
    categories = ["Heard of" ,
                  "Never heard of" ,
                  "Heard of and listened to music EVER" ,
                  "Ever heard of" ,
                  "Ever heard music by",
                  "Dont know if I heard",
                  "Listened to recently",
                  "Heard of and listened to music RECENTLY"]
    reponse=[MISSING_VALUE]

    if value=="__NAME__":
        reponse=[ "HeardOf" ]
    elif value in categories:
        reponse = [ str(categories.index(value)) ]
    else:
        if value!="":
            print value
    return reponse

def cleanOwnArtist(value):
    if "know" in value:
        value = "Dont know if I own"
    categories =["Own none of their music" ,
                 "Own a little of their music" ,
                 "Own a lot of their music" ,
                 "Own all or most of their music" ,
                 "Dont know if I own"
                 ]

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse=categories
    return reponse

def cleanOwnArtistAsValue(value):
    if "know" in value:
        value = "Dont know if I own"
    categories =["Own none of their music" ,
                 "Own a little of their music" ,
                 "Own a lot of their music" ,
                 "Own all or most of their music" ,
                 "Dont know if I own"
                 ]

    reponse=[MISSING_VALUE]

    if value=="__NAME__":
        reponse=[ "OwnArtist" ]
    elif value in categories:
        reponse = [ str(categories.index(value)) ]
    else:
        if value!="":
            print value
    return reponse

def cleanArtist(value):
    categories = ['40', '9', '46', '11', '14', '31', '21', '2', '12', '28', '0', '33', '16', '22', '15', '26', '3', '39', '41', '44', '47', '30', '34', '35', '37', '10', '6', '4', '27', '49', '48', '20', '1', '23', '29', '45', '36', '42', '24', '7', '13', '17', '5', '38', '43', '8', '18', '32', '25', '19']

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print "Unknown value"
            print type(value)
            print value
    if value=="__NAME__":
        reponse= map( lambda x: "Artist"+x, categories)
    return reponse

def cleanTime(value):
    categories =['17', '7', '16', '15', '19', '11', '21', '8', '22', '12', '0', '18', '23', '6', '4', '13', '9']

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse= map( lambda x: "time"+x, categories)
    return reponse

def cleanTrack(value):
    categories =['179', '23', '168', '153', '32', '79', '48', '174', '34', '73', '151', '68', '85', '135', '129', '33', '166', '80', '63', '10', '106', '156', '0', '162', '171', '112', '78', '86', '89', '49', '98', '143', '95', '72', '176', '14', '11', '60', '51', '67', '71', '16', '104', '182', '12', '136', '66', '125', '2', '55', '175', '150', '76', '137', '25', '42', '128', '172', '44', '45', '4', '90', '50', '1', '57', '75', '77', '70', '154', '164', '93', '65', '7', '170', '134', '180', '101', '145', '28', '127', '146', '157', '130', '163', '35', '152', '58', '5', '100', '97', '19', '31', '15', '158', '54', '165', '138', '118', '61', '99', '139', '56', '36', '116', '140', '26', '148', '117', '96', '161', '9', '22', '47', '92', '13', '102', '181', '159', '107', '3', '183', '142', '69', '155', '30', '91', '18', '8', '94', '119', '64', '46', '132', '24', '62', '6', '115', '147', '144', '141', '27', '20', '87', '105', '88', '21', '178', '37', '81', '169', '84', '29', '124', '133', '167', '52', '160', '103', '82', '177', '83', '59', '74', '109', '131', '17', '53', '120', '149', '41', '113', '43', '121', '110', '123', '173', '122', '39', '126', '114', '111', '40', '38', '108']

    reponse = [ "0" for i in range(len(categories))]

    if value in categories:
        reponse[categories.index(value)]="1"
    else:
        if value!="":
            print value
    if value=="__NAME__":
        reponse= map( lambda x: "Track"+x, categories)
    return reponse

def fusionGoodLyrics(col1, col2):
    if col1:
        return col1
    else:
        return col2

def listElementsOfCategory(value):
    global listeCategories
    if value in listeCategories:
        pass
    else:
        listeCategories.append(value)

def isNumber(s):
    try:
        a=float(s)
        return a
    except ValueError:
        return False

def calculateIndividualMeans(newValue, currentTotal, currentIndividualsNumber):
    reponse=(currentTotal, currentIndividualsNumber)
    if newValue!="":
        if newValue!=MISSING_VALUE:
            value = isNumber(newValue)
            if (value or value==0):
                reponse = (currentTotal+value, currentIndividualsNumber+1)
            else:
                print "petite erreure je pense "+newValue
    return reponse





length = 27

users_totals = []
users_individuals = []

for row in reader_users:
    rowResult=[]
    if row[0]!="RESPID":
        rowResult.append(row[0])                 #Respid
        rowResult.append(cleanGender(row[1]))    #Gender
        rowResult.append(addMissingValue(row[2]))

        rowResult+=cleanWork(row[3])
        #rowResult+=cleanWorkAsValues(row[3])
        rowResult+=cleanRegion(row[4])
        #rowResult+=cleanRegionAsValue(row[4])
        rowResult+=cleanMusic(row[5])
        #rowResult+=cleanMusicAsValue(row[5])

        rowResult.append(cleanHours(row[6]))
        rowResult.append(cleanHours(row[7]))

        for i in range(8,length):
            rowResult.append(addMissingValue(row[i]))

        for (index, nombre) in enumerate(rowResult):
            if index>=len(users_totals):
                users_totals.append(0)
                users_individuals.append(0)
            (users_totals[index], users_individuals[index])=calculateIndividualMeans(nombre, users_totals[index], users_individuals[index])

#        for (index, nombre) in enumerate(rowResult):
#            rowResult[index]=replaceMissingValueByMean(nombre, MEANS_USERS, index)

        writer_users.writerow(rowResult)
    else:
        print "First line"
        rowResult.append("User")
        for i in range(1,3):
            rowResult.append(row[i])
        rowResult+=cleanWork("__NAME__")
        #rowResult+=cleanWorkAsValues("__NAME__")
        rowResult+=cleanRegion("__NAME__")
        rowResult+=cleanMusic("__NAME__")
#        rowResult+=cleanRegionAsValue("__NAME__")
#        rowResult+=cleanMusicAsValue("__NAME__")
        for i in range(6,length):
            rowResult.append(row[i])
        print(rowResult)
        writer_users.writerow(rowResult)

print("Means users: ")
means_users = map(truediv, users_totals, users_individuals)
print(means_users)

length = 87

words_totals = []
words_individuals = []

for row in reader_words:
    rowResult=[]
    if row[0]!="Artist":
        rowResult.append(row[0])
        rowResult+=cleanArtist(row[0])
        rowResult.append(row[1])

        rowResult+=cleanHeardOf(row[2])
        #rowResult+=cleanHeardOfAsValue(row[2])
        rowResult+=cleanOwnArtist(row[3])
        #rowResult+=cleanOwnArtistAsValue(row[3])

        rowResult.append(fusionGoodLyrics(row[19],row[49]))

        for i in range(4,length):
            if i>=len(row):
                row.append(MISSING_VALUE)
            if not(i in [19,49]):
                rowResult.append(addMissingValue(row[i]))
    #    cleanHours2(row[5])
        for (index, nombre) in enumerate(rowResult):
            if index>=len(words_totals):
                words_totals.append(0)
                words_individuals.append(0)
            (words_totals[index], words_individuals[index])=calculateIndividualMeans(nombre, words_totals[index], words_individuals[index])

        #for (index, nombre) in enumerate(rowResult):
            #rowResult[index]=replaceMissingValueByMean(nombre, MEANS_WORDS, index)

        writer_words.writerow(rowResult)
    else:
        print "First line"
        rowResult.append(row[0])
        rowResult+=cleanArtist("__NAME__")
        rowResult.append(row[1])
        rowResult+=cleanHeardOf("__NAME__")
        rowResult+=cleanOwnArtist("__NAME__")
        #rowResult+=cleanHeardOfAsValue("__NAME__")
        #rowResult+=cleanOwnArtistAsValue("__NAME__")
        rowResult.append("GoodLyrics")
        for i in range(4,length):
            if not(i in [19,49]):
                rowResult.append(row[i])
        print(rowResult)
        writer_words.writerow(rowResult)

print("Means words: ")
means_words = map(truediv, words_totals, words_individuals)
print(means_words)




for row in reader_train:
    rowResult=[]
    if row[0]!="Artist":
        rowResult.append(row[0])
        rowResult+=cleanArtist(row[0])
        #rowResult.append(row[1])
        rowResult+=cleanTrack(row[1])
        rowResult.append(row[2])
        rowResult.append(row[3])
        #rowResult.append(row[4])
        rowResult+=cleanTime(row[4])
        writer_train.writerow(rowResult)
    else:
        print "First line train"
        rowResult.append(row[0])
        rowResult+=cleanArtist("__NAME__")
        #rowResult.append(row[1])
        rowResult+=cleanTrack("__NAME__")
        rowResult.append(row[2])
        rowResult.append(row[3])
        #rowResult.append(row[4])
        rowResult+=cleanTime("__NAME__")
        writer_train.writerow(rowResult)


print listeCategories

file_in_users.close()
file_out_users.close()
file_in_words.close()
file_out_words.close()
file_in_train.close()
file_out_train.close()



file_out_users=open("../data/users_tmp.csv","r")
file_final_users=open("../data/users_res.csv","w")

file_out_words=open("../data/words_tmp.csv","r")
file_final_words=open("../data/words_res.csv","w")

reader_users = csv.reader(file_out_users)
writer_users = csv.writer(file_final_users)

reader_words = csv.reader(file_out_words)
writer_words = csv.writer(file_final_words)

for row in reader_users:
    for (index, elt) in enumerate(row):
        if elt==MISSING_VALUE:
            row[index]=means_users[index]
    writer_users.writerow(row)

for row in reader_words:
    for (index, elt) in enumerate(row):
        if elt==MISSING_VALUE:
            row[index]=means_words[index]
    writer_words.writerow(row)

file_out_users.close()
file_final_users.close()

file_out_words.close()
file_final_words.close()
