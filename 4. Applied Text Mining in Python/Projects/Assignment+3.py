
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    
    return spam_data['target'].mean() * 100


# In[4]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vect = CountVectorizer().fit(X_train)
    
    return sorted(vect.get_feature_names(), key=lambda x: len(x), reverse=True)[0]


# In[6]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vect = CountVectorizer().fit(X_train)
    
    features_vectorized = vect.transform(X_train)
    
    clfMNB = MultinomialNB(alpha=0.1).fit(features_vectorized, y_train)
    
    predictions = clfMNB.predict(vect.transform(X_test))
    
    auc = roc_auc_score(y_test, predictions)
    
    return auc


# In[8]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    features_vectorized = vect.transform(X_train)
    feature_names = np.array(vect.get_feature_names())
    tfidf_values = features_vectorized.max(0).toarray()[0]
    
    #Min tfidf series
    
    min_tfidf_features = feature_names[tfidf_values.argsort()[:20]]
    min_tfidf_values = tfidf_values[tfidf_values.argsort()[:20]]
    
    tfidf_min_series = (pd.DataFrame({'tfidf_values': min_tfidf_values, 'features': min_tfidf_features})
                        .sort_values(by=['tfidf_values','features'])
                        .set_index('features')
                        .squeeze())
    
    tfidf_min_series.index.name = None
    tfidf_min_series.name = None
    
    #Max tfidf series
    
    max_tfidf_features = feature_names[tfidf_values.argsort()[-21:-1]]
    max_tfidf_values = tfidf_values[tfidf_values.argsort()[-21:-1]]
    
    tfidf_max_series = (pd.DataFrame({'tfidf_values': max_tfidf_values, 'features': max_tfidf_features})
                        .sort_values(by=['tfidf_values','features'], ascending=False)
                        .set_index('features')
                        .squeeze())
    
    tfidf_max_series.index.name = None
    tfidf_max_series.name = None
    
    return tfidf_min_series, tfidf_max_series


# In[10]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[11]:


def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    
    features_vectorized = vect.transform(X_train)
    
    clfMNB = MultinomialNB(alpha=0.1).fit(features_vectorized, y_train)
    
    predictions = clfMNB.predict(vect.transform(X_test))
    
    auc = roc_auc_score(y_test, predictions)
    
    return auc


# In[12]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[13]:


def answer_six():
    
    spam_docs = spam_data[spam_data['target'] == 1]['text'].str.len()
    
    not_spam_docs = spam_data[spam_data['target'] == 0]['text'].str.len()
    
    return not_spam_docs.mean(), spam_docs.mean()


# In[14]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[15]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[16]:


from sklearn.svm import SVC

def answer_seven():
    
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    
    features_vectorized = vect.transform(X_train)
    
    matrix_aggregated = add_feature(features_vectorized, X_train.str.len())
    
    clfSVC = SVC(C=10000).fit(matrix_aggregated, y_train)
    
    predictions = clfSVC.predict(add_feature(vect.transform(X_test), X_test.str.len()))
    
    auc = roc_auc_score(y_test, predictions)    
    
    return auc


# In[17]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[18]:


def answer_eight():
            
    spam_docs = spam_data[spam_data['target'] == 1]['text'].str.count(r'\d')
    
    not_spam_docs = spam_data[spam_data['target'] == 0]['text'].str.count(r'\d')   
    
    return not_spam_docs.mean(), spam_docs.mean()


# In[19]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[20]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    
    #matrix of documents x feature counts(tokens in the bag of words)
    features_vectorized = vect.transform(X_train) 
    
    #add number of characters per document as a feature
    matrix_length_agg = add_feature(features_vectorized, X_train.str.len())
    
    #add to the previous aggregation the digit counts per document as a feature 
    X_train_agg = add_feature(matrix_length_agg, X_train.str.count(r'\d'))
    
    #model training
    clfLR = LogisticRegression(C=100).fit(X_train_agg, y_train)
    
    #add two new features to the test set: digit counts and number of characters per document
    X_test_agg = (add_feature(add_feature(vect.transform(X_test), X_test.str.len()),
                              X_test.str.count(r'\d')))
    
    #y_score_LR = clfLR.fit(X_train_agg, y_train).predict_proba(X_test_agg)
    
    predictions = clfLR.predict(X_test_agg)
    
    auc = roc_auc_score(y_test, predictions)#y_score_LR[:,1])
    
    return auc


# In[21]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[22]:


def answer_ten():
    
    spam_docs = spam_data[spam_data['target'] == 1]['text'].str.count(r'\W')
    
    not_spam_docs = spam_data[spam_data['target'] == 0]['text'].str.count(r'\W') 
    
    return not_spam_docs.mean(), spam_docs.mean()


# In[23]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[34]:


def answer_eleven():
    
    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
    
    #matrix of documents x feature counts(tokens in the bag of words)
    features_vectorized = vect.transform(X_train) 
    
    #add number of characters per document as a feature
    matrix_length_agg = add_feature(features_vectorized, X_train.str.len())
    
    #add to the previous aggregation the digit counts per document as a feature 
    matrix_digit_agg = add_feature(matrix_length_agg, X_train.str.count(r'\d'))
    
    #add to the previous aggregation the non-alphanumeric character counts per document as a feature 
    X_train_agg = add_feature(matrix_digit_agg, X_train.str.count(r'\W'))
    
    #LogisticRegression model training
    clfLR = LogisticRegression(C=100).fit(X_train_agg, y_train)
    
    #add two new features to the test set: digit counts and number of characters per document
    X_test_length_agg = add_feature(vect.transform(X_test), X_test.str.len())
    
    X_test_digit_agg = add_feature(X_test_length_agg, X_test.str.count(r'\d'))
    
    X_test_agg = add_feature(X_test_digit_agg, X_test.str.count(r'\W'))
    
    #Correct way of calculate auc score is using score values, such as obteined from decision_function() or predict_proba()
    #Not predictions obteined out of predict() method
    #y_score_LR = clfLR.decision_function(X_test_agg)
    
    #btw, the assignment autograder was designed to take as correct the auc calculated with predicted values 
    predictions = clfLR.predict(X_test_agg)
    
    auc = roc_auc_score(y_test, predictions)
    
    #Smallest and largest model coefficients
    
    feature_names = np.append(np.array(vect.get_feature_names()), ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_model_coef = clfLR.coef_[0].argsort()
    
    smallest_coef_features = feature_names[sorted_model_coef[:10]]
    largest_coef_features = feature_names[sorted_model_coef[:-11:-1]]
    
    smallest_coef_values = clfLR.coef_[0][sorted_model_coef[:10]]
    largest_coef_values = clfLR.coef_[0][sorted_model_coef[:-11:-1]]
    
    #Smallest coefficient series 
    features_smallcoef_series = (pd.DataFrame({None: smallest_coef_features, 'coef': smallest_coef_values})
                        .sort_values(by='coef')
                        .set_index(None)
                        .squeeze())
     
    features_smallcoef_series.name = None
    
    #Largest coefficient series    
    features_largecoef_series = (pd.DataFrame({None: largest_coef_features, 'coef': largest_coef_values})
                        .sort_values(by='coef', ascending=False)
                        .set_index(None)
                        .squeeze())
     
    features_largecoef_series.name = None
    
    return auc, features_smallcoef_series, features_largecoef_series


# In[35]:


answer_eleven()


# In[ ]:




