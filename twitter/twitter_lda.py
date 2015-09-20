# Author: Alex Perrier <alexis.perrier@gmail.com>
# License: BSD 3 clause
# Python 2.7
'''
This script loads a gensim dictionary and associated corpus
and applies an LDA model.
The documents are timelines of a 'parent' Twitter account.
They are retrieved in their tokenized version from a MongoDb database.

See also the following blog posts

* http://alexperrier.github.io/jekyll/update/2015/09/04/topic-modeling-of-twitter-followers.html
* http://alexperrier.github.io/jekyll/update/2015/09/16/segmentation_twitter_timelines_lda_vs_lsa.html

'''
from gensim import corpora, models, similarities
from pymongo import MongoClient
from time import time
import numpy as np


def connect(DB_NAME):
    client      = MongoClient()
    return client[DB_NAME]

def get_documents(parent):
    condition = {'has_tokens': True, 'is_included': True, 'parent': parent}
    tweets = db.tweets.find(condition).sort("user_id")
    documents = [ { 'user_id': tw['user_id'], 'tokens': tw['tokens']}
                    for tw in tweets  ]
    return documents

# Initialize Parameters
corpus_filename = 'alexip.mm'
dict_filename   = 'alexip.dict'
lda_filename    = 'alexip.lda'
lda_params      = {'num_topics': 40, 'passes': 20, 'alpha': 0.001}

# Connect and get the documents
db              = connect('twitter')
documents       = get_documents('alexip')

print("Corpus of %s documents" % len(documents))

# Load the corpus and Dictionary
corpus = corpora.MmCorpus(corpus_filename)
dictionary = corpora.Dictionary.load(dict_filename)

print("Running LDA with: %s  " % lda_params)
lda = models.LdaModel(corpus, id2word=dictionary,
                        num_topics=lda_params['num_topics'],
                        passes=lda_params['passes'],
                        alpha = lda_params['alpha'])
print()
lda.print_topics()
lda.save(lda_filename)
print("lda saved in %s " % lda_filename)

