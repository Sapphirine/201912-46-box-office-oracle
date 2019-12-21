# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 02:59:04 2019

@author: diego
"""

import numpy as np
import re
import requests
import json
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import collections
import matplotlib.pyplot as plt
import json
import pandas as pd

def get_nrc_data():
    nrc = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    count=0
    emotion_dict=dict()
    with open(nrc,'r') as f:
        for line in f:
            if count < 46:
                count+=1
                continue
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
        cur_array = [vs['neg']*w,
                     vs['pos']*w,
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
        
    return final_array


def get_Data(text_path, N=15):
    
    x = open(text_path, "r").read()
    x = x.split("\n")
    x = [i for i in x if i != '']
    x = [i for i in x if len(i) > 3]
    x = [i.replace("  ", "") for i in x]
    
    n = int(len(x)/N)-1
    x = [x[i:i + n] for i in range(0, len(x), n)]
    x = [''.join(i) for i in x]
    x = [re.split('[?.!]', i) for i in x]
    x = [list(filter(None, i)) for i in x]
    x = [[k for k in i if len(k) > 5] for i in x]
    x = [get_affect(i, len(''.join(i))) for i in x]
    
    #JSON
    array = np.mean(x,axis=0)
    sum_ = np.sum(array)
    d = []
    sent_list = ["Negative","Positive","Compound ","Surprise","Disgust","Joy","Fear","Anger","Anticipation", "Sadness","Trust"]
    for i in range(11):
        a = dict()
        a['area'] = sent_list[i]
        a['value'] = (array[i]/sum_)*5
        d.append(a)
    
    with open('data.json', 'w') as fp:
        json.dump([d], fp)
    
    #CSV
    csv_val = [i[2] for i in x]
    blobs = ['blob_1','blob_2','blob_3','blob_4','blob_5',
             'blob_6','blob_7','blob_8','blob_9','blob_10',
             'blob_11','blob_12','blob_13','blob_14','blob_15','blob_16']
    
    df = pd.DataFrame()
    df['Blob'] = blobs
    df['Values'] = csv_val
    df.to_csv('circlebar.csv', encoding='utf-8',index=False)
    
    return x




