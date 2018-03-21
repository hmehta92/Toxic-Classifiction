
import pandas as pd
import numpy as np
import regex
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
porter = PorterStemmer()
lmtzr = WordNetLemmatizer()
stop = set(stopwords.words('english'))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../inputtest.csv")
train=train.drop(["id"],axis=1)
test=test.drop(["id"],axis=1)
# define function for checking comment_text
def check(data):
    text=data["comment_text"].values
    single_character=[]
    index=[]
    for i,n in enumerate(text):
        if len(n)<=6:
            single_character.append(n)
            index.append(i)
    return single_character,index       

train=train[~train["comment_text"].isin(check(train)[0])]

test[test["comment_text"].isin(check(test)[0])]="unknown"
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
        lemmed = [lmtzr.lemmatize(word) for word in tokens]
        filtered_tokens = [w for w in lemmed if regex.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
train["tokens"]=train["comment_text"].apply(lambda x:tokenize(x))
test["tokens"]=test["comment_text"].apply(lambda x:tokenize(x))
train["len"]=train["tokens"].apply(lambda x:len(x))
test["len"]=test["tokens"].apply(lambda x:len(x))
train["comment_text"]=train["tokens"].apply(lambda x:" ".join(x))
test["comment_text"]=test["tokens"].apply(lambda x:" ".join(x))
# splitting dataset into train and validation set
x_train=train.sample(frac=0.8)
x_validation=train.drop(x_train.index)
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import mean_squared_error as mse 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer
vect_word = TfidfVectorizer(max_features=90000, lowercase=True, analyzer='word',min_df=0.0001,max_df=0.999,
                        stop_words= 'english',ngram_range=(1,2),dtype=np.float32,sublinear_tf=1)
word=vect_word.fit(np.append(train["comment_text"].values,test["comment_text"].values))
tr_vect = word.transform(x_train['comment_text'])
validation_vect=word.transform(x_validation['comment_text'])
ts_vect = word.transform(test['comment_text'])
prediction1=[]
model1=LR(C=4,max_iter=300,verbose=1,tol=.0001,solver="lbfgs")
for i in columns:
    y=x_train[i]
    model1.fit(tr_vect,y)
    pred=model1.predict_proba(validation_vect)
    pred1=model1.predict_proba(ts_vect)
    value=auc(x_validation[i],pred[:,1])
    value1=mse(x_validation[i],pred[:,1])
    auc_max.update({i:value})
    print("auc score is: {} ".format(value))
    print("error :{}".format(value1))
    prediction1.append(pred1[:,1])
result1=pd.DataFrame(np.transpose(prediction1),columns=columns)
submission1=pd.concat([index,result1],axis=1)
submission1.to_csv("result_LR.csv",index=False)
