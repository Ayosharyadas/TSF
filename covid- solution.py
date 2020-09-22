#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


covid=pd.read_csv("D:\data1\\covid19_tweets.csv")


# In[5]:


import seaborn as sns


# In[ ]:





# In[6]:


covid.head(10)


# In[7]:


covid.dtypes


# In[8]:


covid.describe()


# In[9]:


covid.dtypes=="object"
a = covid.dtypes[covid.dtypes == "object"].index
a
covid[a].describe()


# In[10]:


covid.dtypes=="bool"
b= covid.dtypes[covid.dtypes == "bool"].index
b
covid[b].describe()


# In[11]:


covid.columns


# In[12]:


covid['hashtags'].unique()


# In[13]:


covid['source'].unique()


# In[14]:


covid['user_location'].unique()


# In[15]:


covid['user_description'].unique()


# In[16]:


covid['user_verified'].unique()


# In[17]:


covid.isnull().sum()


# In[18]:


covid.shape


# In[19]:


np.where(covid["user_followers"]==max(covid["user_followers"]))


# In[20]:


row_index=np.where(covid["user_followers"]==max(covid["user_followers"]))


# In[21]:


covid.iloc[row_index]


# In[22]:


covid=covid.drop(['is_retweet'],axis=1) #all column have same data it has no impact on our dataset


# In[23]:


covid.head(3)


# In[24]:


covid['hashtags'].value_counts()


# In[25]:


covid['source'].value_counts()


# In[26]:


covid['user_description'].value_counts()


# In[27]:


covid['user_location'].value_counts()


# In[28]:


#PERFORM IMPUTATION


# In[29]:


covid.isnull().sum()/len(covid)  #check for percentage of na values


# In[30]:


covid_hastag_mode = covid['hashtags'].value_counts().index[0] # performing imputation


# In[31]:


covid['hashtags'].fillna(covid_hastag_mode,inplace=True)


# In[32]:


covid_user_location_mode = covid['user_location'].value_counts().index[0]


# In[33]:


covid['user_location'].fillna(covid_user_location_mode,inplace=True)


# In[34]:


covid_user_description_mode = covid['user_description'].value_counts().index[0]


# In[35]:


covid['user_description'].fillna(covid_user_description_mode,inplace=True)


# In[36]:


covid_source_mode = covid['source'].value_counts().index[0]


# In[37]:


covid['source'].fillna(covid_source_mode,inplace=True)


# In[38]:


covid.isnull().sum()/len(covid)# again check for percentage of missing value


# In[39]:


#EDA PROCESS


# In[40]:


#for eda propose we take sample


# In[41]:


from random import sample


# In[42]:


rindex = np.array(sample(range(len(covid)),100))


# In[43]:


cov= covid.iloc[rindex]


# In[44]:


import datetime as dt


# In[45]:


cov['date']=pd.to_datetime(cov['date'])
cov['year'] = cov['date'].dt.year
cov['month'] = cov['date'].dt.month
cov['day'] =cov['date'].dt.day


# In[46]:


cov.head(5)


# In[47]:


location= cov.groupby(['user_location'])['hashtags'].nunique().sort_values(ascending=False)


# In[48]:


location


# In[49]:


m1 = cov.groupby(['hashtags','user_location'])


# In[50]:


m2 = pd.DataFrame(m1)


# In[51]:


m2 


# In[52]:


source= cov.groupby(['source'])['hashtags'].nunique().sort_values(ascending=False)


# In[53]:


source


# In[54]:


m3 = cov.groupby(['hashtags','source'])


# In[55]:


m4 = pd.DataFrame(m3)


# In[56]:


m4


# In[57]:


description= cov.groupby(['user_description'])['hashtags'].nunique().sort_values(ascending=False)


# In[58]:


description


# In[59]:


m5 = cov.groupby(['hashtags','user_description'])


# In[60]:


m6 = pd.DataFrame(m5)
m6


# In[61]:


plt.figure(1)
plt.subplot(221)
cov['hashtags'].value_counts(normalize=True).plot.bar(figsize=(20,10),title = "hashtags")


# In[62]:


covid['source'].value_counts()[:20].plot(kind='barh')


# In[63]:


sns.countplot(x='user_verified',data=covid,palette='hls')


# In[64]:


#text cleansing


# In[65]:


import nltk
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


# In[66]:


from collections import Counter


# In[67]:


nltk.download('punkt')


# In[68]:


def text_clean(text): 
    # Replacing the repeating pattern of &#039;
    pattern_remove = text.str.replace("&#039;", "")
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    # Replacing Two or more dots with one
    cleannum = multiw_remove.str.replace(r'\.{2,}', ' ')
    # cleaning numbers
    dataframe = cleannum.str.replace('\d+', '')
    
    return dataframe


# In[69]:


covid['text']=text_clean(covid['text'])  # cleaned review 
covid['user_description']=text_clean(covid['user_description'])  # cleaned review 

covid


# In[70]:


review_text = [text for text in covid['text']]


# In[71]:


review_text 


# In[72]:


def identify_tokens(row):
    review = row['text']
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


# In[73]:


covid['words'] = covid.apply(identify_tokens, axis=1)


# In[74]:


covid['words'] 


# In[75]:


def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

covid['stemwords'] = covid.apply(stem_list, axis=1)


# In[76]:


covid


# In[81]:



from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[80]:


pip install wordcloud


# In[95]:



wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


#sentimental analysis


# In[85]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# In[87]:


covid['scores'] = covid['text'].apply(lambda text: sid.polarity_scores(text))


# In[88]:


covid.head(5)


# In[90]:


covid['compound']  = covid['scores'].apply(lambda score_dict: score_dict['compound'])


# In[91]:


covid.head(5)


# In[92]:


covid['comp_score'] = covid['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')


# In[93]:



covid.head(5)


# In[94]:


covid['comp_score'].value_counts()


# In[ ]:


covid['text'] = covid[covid.comp_score ==pos]
neg_pos
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')


# In[106]:


neg_tweets = covid[covid.compound < 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')


# In[107]:


wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[108]:


pos_tweets = covid[covid.compound > 0]
pos_string = []
for t in neg_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')


# In[109]:


wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:




