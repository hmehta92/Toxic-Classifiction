# Toxic-Classifiction
Introduction:
The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

Objective:
Build a Predictive model to classify the tweets among different Categories as a online challenge on Kaggle. Challenge is to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models.

Data Description:
A large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:
toxic
severe_toxic
obscene
threat
insult
identity_hate
Build a model which predicts a probability of each type of toxicity for each comment.

Methodology:
Used couple of methods to do the word vectorization and to create models. 
I first cleaned the data by removing the stop words and used NLTK tokenizer, stemming and lemmatization for creating the tokens from the available corpes. Used tfidf vectorizer for creating word and character vectors. Tunned different hyperparameters of tfidf vectorizer. Stacked the word and character vectors and used Logistic Regression model.
Since, Tfidf does not capture the semantic, so i used Glove ( Global vectors for word representation) word embedding technique and used Keras for building the LSTM model. 
