# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:00:50 2019

@author: diego
"""

import pandas as pd
import operator
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict
from graphframes import *
from pyspark.sql.functions import desc
import pyspark.sql.functions as f
import re
import requests
import json
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
from google.cloud import storage
from pyspark.ml.linalg import Vectors


nltk.download('punkt')

conf = SparkConf().setMaster('local[*]')
sc = SparkContext.getOrCreate(conf=conf)

def split(txt, seps):
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


def get_nrc_data():
    client = storage.Client()
    bucket = client.get_bucket('project-bucket-boo')
    blob = bucket.get_blob('NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt')
    your_file_contents = blob.download_as_string()
    count=0
    emotion_dict=dict()
    all_lines = list()
    for line in your_file_contents.decode("utf-8").strip().split('\n')[35:]:
        line = line.strip().split('\t')
        if int(line[2]) == 1:
            if emotion_dict.get(line[0]):
                emotion_dict[line[0]].append(line[1])
            else:
                emotion_dict[line[0]] = [line[1]]
    return emotion_dict


def emotion_analyzer(text,emotion_dict=get_nrc_data()):
    #Set up the result dictionary
    emotions = {x for y in emotion_dict.values() for x in y}
    emotion_count = dict()
    for emotion in emotions:
        emotion_count[emotion] = 0

    #Analyze the text and normalize by total number of words
    total_words = len(text.split())
    for word in text.split():
        if emotion_dict.get(word):
            for emotion in emotion_dict.get(word):
                emotion_count[emotion] += 1
    return emotion_count


def get_affect(text, length):
    analyzer = SentimentIntensityAnalyzer()
#     sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    final_array = [0,0,0,0,0,0,0,0,0,0,0] 
    for line in text:
        #sentence = sent_detector.tokenize(line.strip())
        vs = analyzer.polarity_scores(line)
        ea = emotion_analyzer(line)
        w = len(line)/length
        cur_array = [\
#             vs['neg']*w,
#                      vs['pos']*w,
                     vs['compound']*w,
                     ea['surprise']*w,
                     ea['disgust']*w,
                     ea['joy']*w,
                     ea['fear']*w,
                     ea['anger']*w,
                     ea['anticipation']*w,
                     ea['sadness']*w,
                     ea['trust']*w]
        final_array = [x + y for x, y in zip(final_array, cur_array)]
#         final_array = np.array(final_array)
#         final_array = Vectors.dense(final_array)
    return final_array


def getData(sc, filename, N = 15):

    
    #split by lines, filter empty rows, filter elements with less than 3 characters
    #and take out extra spaces
    data = sc.textFile(filename)
    data  = data.map(lambda line: line.split("\t"))\
                .filter(lambda x: x[0] is not u'')\
                .filter(lambda x: len(x[0]) > 3)\
                .map(lambda x: x[0].replace("  ", ""))
    
    #split script into n equal pieces
    data = data.collect()
    n = int(len(data)/N)-1
    data = sc.parallelize([data[i:i + n] for i in range(0, len(data), n)])
    
    #getting full sentences and affect for each blob
    data = data.map(lambda x : ''.join(x))\
               .map(lambda x : re.split('[?.!-]', x))\
               .map(lambda x : list(filter(None, x)))\
               .map(lambda x : [k for k in x if len(k) > 5])\
               .map(lambda x : get_affect(x, len(''.join(x))))

    return data


#Final Cleaning Stage
    
filename = "gs://project-bucket-boo/all_titles.txt"
titles = sc.textFile(filename)


movies_list = titles.collect()[0].split(",")
movies_analysis = []
for title in movies_list:
    filename = "gs://project-bucket-boo/input/%s"% (title,)
    try:
        data = getData(sc, filename)
        movies_analysis.append(data.collect())
    except:
        print(title)
        pass
    
f_ = ['.DS_Store','starshiptroopers.txt','apocalypsenow.txt','youvegotmail.txt','stingthe.txt','peggysuegotmarried.txt','Action_BO.pkl']

for i in range(len(movies_list)):
    if movies_list[i] in f_:
        print (i,movies_list[i])

#Save as Spark pickle
sc.parallelize(final_movie_list).saveAsPickleFile("gs://project-bucket-boo/output_2/movies_analysis.pickle")







