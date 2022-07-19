# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:11:26 2022

@author: Shah
"""

import os
import pandas as pd
import re
import numpy as np
from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional
from tensorflow.keras import Input,Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#STEP 1 Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')


#%%STEP 2 Data Inspection
df.info()
df.describe().T
df.head()

df.duplicated().sum()
df.isna().sum()

print(df['review'][4])
print(df['review'][10])

# Symbol and HTML Tags have to be removed
#%%STEP 3 Data Cleaning
# test = df['review'][20]
# print('_________BEFORE___________')
# print(test)
# print('_________AFTER___________')
# #re.sub('<.*?>','',test) #. = any character
# #print(re.sub('[^A-zA-Z]',' ',test).lower())

# test = re.sub('<.*?>','',test)
# test = re.sub('[^a-zA-Z]',' ',test).lower().split() # add split to seperate
# print(test)

review = df['review']
sentiment = df['sentiment']

for index, text in enumerate(review):  #loop it to do it for all data
    review[index] = re.sub('<.*?>','',text)
    review[index] = re.sub('[^a-zA-Z]',' ',text).lower().split() #test become text
    #test become text, # to remove html tags
    #anythng withing the <> will be removed including <>
    #? to tell re dont be greedy so it wont capture everything
    #from the first < to the last > in the document

review_backup = review.copy()
sentiment_backup = sentiment.copy()

#%%STEP 4 Features Selection

vocab_size=10000
oov_token='<OOV>'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) #to learn
word_index=tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

review_int=tokenizer.texts_to_sequences(review) #to convert to number
review_int[100] #to check all convert to number

# length_review = []
# for i in range(len(review.int)):
#     length_review.append(len(review_int[i]))
#     # print(len(review[i]))



# np.median(length_review)

max_len=np.median([len(review_int[i])for i in range(len(review_int))])


padded_review=pad_sequences(review_int,
                            maxlen=int(max_len),
                            padding='post',
                            truncating='post')
# Y target

ohe=OneHotEncoder(sparse=False)
sentiment=ohe.fit_transform(np.expand_dims(sentiment,axis=-1))


X_train,X_test,y_train,y_test=train_test_split(padded_review,
                                               sentiment,
                                               test_size=0.3,
                                               random_state=(123))

#%%STEP 5 Data Preprocessing

from tensorflow.keras.utils import plot_model

# X_train=np.expand_dims(X_train,axis=-1)
# X_test=np.expand_dims(X_test,axis=-1)

input_shape=np.shape(X_train)[1:]
out_dim = 128

model=Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,out_dim))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.summary()

plot_model(model,show_shapes=(True))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])
#%%model training

#callbacks

LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

hist = model.fit(X_train,y_train,
          epochs=5,
          callbacks=[tensorboard_callback],
          validation_data=(X_test,y_test))

#%%model analysis

from sklearn.metrics import classification_report 

y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test, axis=1)

print(classification_report(y_actual,y_pred))

#%%model saving

#TOKENIZER
import json

TOKENIZER_SAVE_PATH=os.path.join(os.getcwd(),'model','tokenizer.json')

token_json = tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH,'w')as file:
    json.dump(token_json,file)


# OHE
import pickle
OHE_SAVE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')

with open(OHE_SAVE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#MODEL
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')
model.save(MODEL_SAVE_PATH)
