# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:26:59 2019

@author: diego
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam

data = sc.pickleFile('gs://project-bucket-boo/output/movies_analysis.pickle')

movies_analysis = data.collect()

filename = "gs://project-bucket-boo/all_tiers.txt"
tiers = sc.textFile(filename)

movies_tiers = tiers.collect()[0].split(",")

#Final Cleaning before training

remove_indices = []
for i in range(691):
    if movies_analysis[i] == []:
        remove_indices.append(i)

movies_analysis = [i for j, i in enumerate(movies_analysis) if j not in remove_indices]
movies_tiers = [i for j, i in enumerate(movies_tiers) if j not in remove_indices]

remove_indices = []
for i in range(645):
    if len(movies_analysis[i]) != 16:
        remove_indices.append(i)

movies_analysis = [i for j, i in enumerate(movies_analysis) if j not in remove_indices]
movies_tiers = [i for j, i in enumerate(movies_tiers) if j not in remove_indices]

data = np.array(movies_analysis).reshape((len(movies_analysis), 16, 11))


#KERAS LSTM MODEL

model = tf.keras.Sequential()
model.add(LSTM(100, input_shape=(16, 11)))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

#Training Stage
X = data
Y = np.array(movies_tiers)
Y = to_categorical(Y)

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.2)

batch_size = 5
model.fit(X_train, Y_train, batch_size =batch_size, epochs = 30,  verbose = 2)


##SAVE MODEL OUT OF GCP
# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

##SAVE MODEL IN GCP
from google.cloud import storage
client = storage.Client()
bucket = client.get_bucket('project-bucket-boo')
blob2 = bucket.blob('gs://project-bucket-boo/model.json')
blob2.upload_from_filename(filename='model.json')
blob2 = bucket.blob('gs://project-bucket-boo/model.h5')
blob2.upload_from_filename(filename='model.h5')