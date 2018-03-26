

import pandas as pd
import numpy as np
import os
os.chdir("C:/Users/Harshit Mehta/Desktop/austin/kaggle/project 8/train")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
index=test["id"]
train=train.drop(["id"],axis=1)
test=test.drop(["id"],axis=1)
columns=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
import keras
import tensorflow
import regex
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex1 = regex.compile('[' +regex.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex1.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if regex.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
train["tokens"]=train["comment_text"].apply(lambda x:tokenize(x))
test["tokens"]=test["comment_text"].apply(lambda x:tokenize(x))
train["comment_text"]=train["tokens"].apply(lambda x:" ".join(x))
test["comment_text"]=test["tokens"].apply(lambda x:" ".join(x))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
texts=np.append(train["comment_text"].values,test["comment_text"].values)
maxlen = 400
max_words = 50000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
x_train=data[:130000]
x_valid=data[130000:159569]
x_test=data[159569:]
embeddings_index = {}
f = open('glove.6B.200d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 200
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense,LSTM
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
#model.add(Flatten())
model.add(LSTM(100))
#model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='Adagrad',
              loss='binary_crossentropy',
              metrics=['acc'])
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
predictions=[]
for i in columns:
    model.fit(x_train, train[i][:130000],epochs=7,batch_size=256,validation_data=(x_valid, train[i][130000:159569]))
    pred=model.predict_proba(x_test)
    predictions.append(pred)
A=np.hstack((predictions[0],predictions[1],predictions[2],predictions[3],predictions[4],predictions[5]))
result=pd.DataFrame(A,columns=columns)
subm=pd.concat([index,result],axis=1)
subm.to_csv("result_KR.csv",index=False)

