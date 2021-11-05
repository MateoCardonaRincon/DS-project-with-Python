
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[33]:


import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()


# In[34]:


# If you would like to work with the novel in nltk.Text format you can use 'text1'
nltk.download('punkt', 'wordnet', 'averaged_perceptron_tagger')


# In[35]:


moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw))# or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[ ]:


def answer_one():
    
    
    return len(set(nltk.word_tokenize(moby_raw)))/len(nltk.word_tokenize(moby_raw))

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[37]:


def answer_two(): 
    
    return 100*len([w for w in text1 if w in ['Whale', 'whale']])/len(text1)

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[ ]:


def answer_three():
    
    dist = nltk.FreqDist(text1)
    
    return [(w, dist[w]) for w in sorted(dist, key=dist.get, reverse=True)][:20]

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[ ]:


def answer_four():
    
    dist = nltk.FreqDist(text1)
    
    
    return sorted([w for w in set(text1) if (dist[w] > 150) and (len(w) > 5)])

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[ ]:


def answer_five():
    
    
    
    return sorted([(w, len(w)) for w in set(text1)], key = lambda x: x[1], reverse=True)[0]

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[ ]:


def answer_six():
    
    dist = nltk.FreqDist(text1)
    
    return [(dist[w], w) for w in sorted(dist, key=dist.get, reverse=True) if w.isalpha() and dist[w] > 2000]

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[28]:


def answer_seven():
    
    sents = nltk.sent_tokenize(moby_raw)
    
    avg_tokens_per_sentence = sum([len(nltk.word_tokenize(w)) for w in sents])/len(sents)
   
    return avg_tokens_per_sentence

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[43]:


def answer_eight():
    
    tokens = nltk.word_tokenize(moby_raw)
    
    pos = list(zip(*nltk.pos_tag(tokens)))[1]
    
    pos_dist = nltk.FreqDist(pos)
    
    pos_dist_sorted = [(p, pos_dist[p]) for p in sorted(pos_dist, key=pos_dist.get, reverse=True)[:5]]

    return pos_dist_sorted

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[3]:


from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[21]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):

    recomendations = []
    
    for entry in entries:
        
        ground_jaccard_distance = 1
        recomendation = entry
        word_list = filter(lambda x: x[0] == entry[0], correct_spellings)
        
        for word in word_list:
            
            jd = nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), set(nltk.ngrams(word, n=3)))
            
            if jd < ground_jaccard_distance:
                ground_jaccard_distance = jd
                recomendation = word
                
        recomendations.append(recomendation)

    
    return recomendations
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[25]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    recomendations = []
    
    for entry in entries:
        
        ground_jaccard_distance = 1
        recomendation = entry
        word_list = filter(lambda x: x[0] == entry[0], correct_spellings)
        
        for word in word_list:
            
            jd = nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)), set(nltk.ngrams(word, n=4)))
            
            if jd < ground_jaccard_distance:
                ground_jaccard_distance = jd
                recomendation = word
                
        recomendations.append(recomendation)
    
    return recomendations
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[27]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    recomendations = []
    
    for entry in entries:
        
        min_edit_distance = len(entry)
        recomendation = entry
        word_list = filter(lambda x: x[0] == entry[0], correct_spellings)
        
        for word in word_list:
            
            ed = nltk.edit_distance(entry, word, transpositions=True)
            
            if ed < min_edit_distance:
                min_edit_distance = ed
                recomendation = word
                
        recomendations.append(recomendation)
    
    return recomendations
    
answer_eleven()


# In[ ]:




