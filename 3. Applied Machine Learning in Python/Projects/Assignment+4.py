
# coding: utf-8

# ---
#
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
#
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
#
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)).
#
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
#
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
#
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
#
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
#
# ___
#
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
#
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
#
# <br>
#
# **File descriptions** (Use only this data for training your model!)
#
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates.
#      Note: misspelled addresses may be incorrectly geolocated.
#
# <br>
#
# **Data fields**
#
# train.csv & test.csv
#
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#
# train.csv only
#
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction]
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
#
#
# ___
#
# ## Evaluation
#
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
#
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).
#
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
#
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
#
# Example:
#
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#
# ### Hints
#
# * Make sure your code is working before submitting it to the autograder.
#
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
#
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question.
#
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
#
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[ ]:

###Assignment workflow###
# Data processing: get_dummies -> 265406 features - Not viable
# Do feature ingeneering instead. Select around 5 features which fit the better classifier. Use latlons dataset
# Using numerical features is enough for this assignment
#
# Use a classifier that admits predict_proba() (RandomForestClassifier, LogisticRegrssion, GradientBoosted DT, etc.)
# Split train.csv dataset in train_train set (training set) and train_test set (validation set)
# Fit model on train_train set
# Use cross validation on train_train and train_test sets to look for underfitting or overfitting
# If it's necessary, use GridSearchCV for parameters tuning
# Verify AUC such that be over 0.75 for test.csv data set (test set - unseen data)
# Apply predict_proba() on test set
#
# Retuns a series whose values are the second component (prodiction probabilities)
# from the predict_proba() output with 'ticket_id' as indexes


# In[1]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[2]:

pd.set_option('display.max_columns', None)

train_set = pd.read_csv('readonly/train.csv', engine='python')
test_set = pd.read_csv('readonly/test.csv', engine='python')
addresses = pd.read_csv('readonly/addresses.csv')
lats_lons = pd.read_csv('readonly/latlons.csv')


# In[3]:

ticket_latlon = addresses.merge(lats_lons, how='inner', on='address')[
    ['ticket_id', 'lat', 'lon']]

relevant_features = ['ticket_id', 'discount_amount',
                     'judgment_amount', 'compliance']


# In[4]:

train_cleaned = train_set.copy()[relevant_features]
training = train_cleaned.merge(ticket_latlon, how='inner', on='ticket_id')

training.dropna(axis=0, inplace=True)
training.head(1)


# In[5]:

test_cleaned = test_set.copy()[relevant_features[:-1]]
testing = test_cleaned.merge(ticket_latlon, how='inner', on='ticket_id')

test_ids = testing['ticket_id']
testing.drop('ticket_id', axis=1, inplace=True)
testing.head(1)


# In[7]:

X = training.iloc[:, [1, 2, 4, 5]]
y = training['compliance']


# In[41]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Negative class (0) is most frequent
dummy_majority = DummyClassifier(
    strategy='most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)

print('Dummy C accuracy', dummy_majority.score(X_test, y_test))
print('Cross-validation (AUC)', cross_val_score(dummy_majority,
                                                X_test, y_test, cv=5, scoring='roc_auc').mean())


# In[68]:


# In[69]:

# LogisticRegression Classifier


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for this_C in [0.001, 0.01, 0.1, 1, 10, 100]:

    logireg = LogisticRegression(C=this_C).fit(X_train_scaled, y_train)
    y_score_lr = logireg.decision_function(X_test_scaled)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    accuracy_lr = logireg.score(X_test_scaled, y_test)

    print("LogisticRegression with C = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(
        this_C, accuracy_lr, roc_auc_lr))
    print('Cross-validation (AUC)', cross_val_score(logireg,
                                                    X_test_scaled, y_test, cv=5, scoring='roc_auc').mean())


# In[70]:

# Naive Bayes Classifier


gaussnb = GaussianNB().fit(X_train, y_train)

print('GaussianNB accuracy', gaussnb.score(X_test, y_test))
print('Cross-validation (AUC)', cross_val_score(gaussnb,
                                                X_test, y_test, cv=5, scoring='roc_auc').mean())


# In[71]:

# RandomForest Classifier


RF = RandomForestClassifier().fit(X_train, y_train)

print('RandomForest accuracy', RF.score(X_test, y_test))
print('Cross-validation (AUC)', cross_val_score(RF,
                                                X_test, y_test, cv=5, scoring='roc_auc').mean())


# In[72]:


GBDT = GradientBoostingClassifier().fit(X_train, y_train)
print('GradientBoosted Decision tree  accuracy', GBDT.score(X_test, y_test))
print('Cross-validation (AUC)', cross_val_score(GBDT,
                                                X_test, y_test, cv=5, scoring='roc_auc').mean())


# In[86]:

GBDT = GradientBoostingClassifier(max_depth=5)
grid_values = {'learning_rate': [0.1, 0.15, 0.2],
               'n_estimators': [300, 200, 100]}

# alternative metric to optimize over grid parameters: AUC
grid_GBDT_auc = GridSearchCV(GBDT, param_grid=grid_values, scoring='roc_auc')
grid_GBDT_auc.fit(X_train, y_train)
y_decision_fn_scores_auc = grid_GBDT_auc.decision_function(X_test)

print('GradientBoostingClassifier')
print('Train set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
print('Grid best parameter (max. AUC): ', grid_GBDT_auc.best_params_)
print('Grid best score (AUC): ', grid_GBDT_auc.best_score_)


# In[95]:

GBDT = GradientBoostingClassifier(
    learning_rate=0.2, n_estimators=150, max_depth=5)
GBDT.fit(X_train, y_train)

y_train_decision_fn_scores_auc = GBDT.decision_function(X_train)
y_test_decision_fn_scores_auc = GBDT.decision_function(X_test)


print('GradientBoostingClassifier')
print('Train set AUC: ', roc_auc_score(
    y_train, y_train_decision_fn_scores_auc))
print('Test set AUC: ', roc_auc_score(y_test, y_test_decision_fn_scores_auc))


# In[ ]:

# Best results:

# GradientBoostingClassifier - learning_rate = 0.2, n_estimators = 300
# Train set AUC:  0.815269022662
# Test set AUC:  0.775155879689

# GradientBoostingClassifier - learning_rate = 0.2, n_estimators = 200
# Train set AUC:  0.802614689192
# Test set AUC:  0.773253503226
