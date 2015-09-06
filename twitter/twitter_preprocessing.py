'''
This script

* Loads documents as aggregation of tweets stored in a MongoDB collection
* Cleans up the documents
* Creates a dictionary and corpus that can be used to train an LDA model
* Training of the LDA model is not included but follows:
  lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=100)

Author: Alex Perrier
Python 2.7
'''

import langid
import nltk
import re
import time
from collections import defaultdict
from configparser import ConfigParser
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from pymongo import MongoClient
from string import digits


def filter_lang(lang, documents):
    doclang = [  langid.classify(doc) for doc in documents ]
    return [documents[k] for k in range(len(documents)) if doclang[k][0] == lang]

# connect to the MongoDB
client      = MongoClient()
db          = client['twitter']

# Load documents and followers from db
# Filter out non-english timelines and TL with less than 2 tweets
documents    = [tw['raw_text'] for tw in db.tweets.find()
                    if ('lang' in tw.keys()) and (tw['lang'] in ('en','und'))
                        and ('n_tweets' in tw.keys()) and (tw['n_tweets'] > 2) ]

#  Filter non english documents
documents = filter_lang('en', documents)
print("We have " + str(len(documents)) + " documents in english ")

# Remove urls
documents = [re.sub(r"(?:\@|http?\://)\S+", "", doc)
                for doc in documents ]

# Remove documents with less 100 words (some timeline are only composed of URLs)
documents = [doc for doc in documents if len(doc) > 100]

# tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
documents = [ tokenizer.tokenize(doc.lower()) for doc in documents ]

# Remove stop words
stoplist_tw=['amp','get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via',
            'one','com','new','like','great','make','top','awesome','best',
            'good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','best','really','see','want','nice',
            'while','know']

unigrams = [ w for doc in documents for w in doc if len(w)==1]
bigrams  = [ w for doc in documents for w in doc if len(w)==2]

stoplist  = set(nltk.corpus.stopwords.words("english") + stoplist_tw
                + unigrams + bigrams)
documents = [[token for token in doc if token not in stoplist]
                for doc in documents]

# rm numbers only words
documents = [ [token for token in doc if len(token.strip(digits)) == len(token)]
                for doc in documents ]

# Lammetization
# This did not add coherence ot the model and obfuscates interpretability of the
# Topics. It was not used in the final model.
#   from nltk.stem import WordNetLemmatizer
#   lmtzr = WordNetLemmatizer()
#   documents=[[lmtzr.lemmatize(token) for token in doc ] for doc in documents]

# Remove words that only occur once
token_frequency = defaultdict(int)

# count all token
for doc in documents:
    for token in doc:
        token_frequency[token] += 1

# keep words that occur more than once
documents = [ [token for token in doc if token_frequency[token] > 1]
                for doc in documents  ]

# Sort words in documents
for doc in documents:
    doc.sort()

# Build a dictionary where for each document each word has its own id
dictionary = corpora.Dictionary(documents)
dictionary.compactify()
# and save the dictionary for future use
dictionary.save('alexip_followers_py27.dict')

# We now have a dictionary with 26652 unique tokens
print(dictionary)

# Build the corpus: vectors with occurence of each word for each document
# convert tokenized documents to vectors
corpus = [dictionary.doc2bow(doc) for doc in documents]

# and save in Market Matrix format
corpora.MmCorpus.serialize('alexip_followers_py27.mm', corpus)
# this corpus can be loaded with corpus = corpora.MmCorpus('alexip_followers.mm')


