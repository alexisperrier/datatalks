# Author: Alex Perrier <alexis.perrier@gmail.com>
# License: BSD 3 clause
# Python 2.7 / 3.4

'''
A wrapper script on scikit-learn TruncatedSVD and k-means
See http://scikit-learn.org/stable/auto_examples/document_clustering.html

Given a set of documents (twitter timelines), this script

* Tokenizes the raw text
* Applies Latent Semantic Allocation (aka TruncatedSVD)
* Normalizes the frequency Matrix
* Applies K-means to segment the documents

In our case, the documents are already tokenized and cleaned up timelines
of a 'parent' Twitter account. They are retrieved in their tokenized version
from a MongoDb database.

See
* https://github.com/alexperrier/datatalks/blob/master/twitter/twitter_tokenize.py

See also the following blog posts

* http://alexperrier.github.io/jekyll/update/2015/09/04/topic-modeling-of-twitter-followers.html
* http://alexperrier.github.io/jekyll/update/2015/09/16/segmentation_twitter_timelines_lda_vs_lsa.html
'''

from pymongo import MongoClient
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt
import numpy as np
import operator


def connect(DB_NAME):
    client      = MongoClient()
    return client[DB_NAME]

def get_documents(parent):
    '''
        Retrieves the tokenized version of the timelines,
        followers of a 'parent' account
        that we choose to include in the corpus: is_included: True
    '''
    condition   = {'has_tokens': True, 'is_included': True, 'parent': parent}
    tweets      = db.tweets.find(condition).sort("user_id")
    documents   = [ { 'user_id': tw['user_id'], 'tokens': tw['tokens']}
                        for tw in tweets  ]
    return documents

def display_topics(svd, terms, n_components, n_out = 7, n_weight = 5, topic = None):
    '''
        This displays a weight measure of each topic (dimension)
        and the 'n_out' first words of these topics.
        n_weight is the number of words used to calculate the weight
        Input:
            svd: the TruncatedSVD model that has been fitted
            terms: the list of words
            n_components: The reduced dimension
            topic: by default prints all topics in the SVD, if topic (int) given
                    prints only the weight and words for that topic
            n_out: Number of words per topic to display
            n_weight: Number of words to average on to calculate the weight
                    of the topic. The smaller, the more spread bwteen the topic
                    relative weights
    '''

    if topic is None:
        for k in range(n_components):
            idx = {i:abs(j) for i, j in enumerate(svd.components_[k])}
            sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
            weight = np.mean([ item[1] for item in sorted_idx[0:n_weight] ])
            print("T%s)" % k, end =' ')
            for item in sorted_idx[0:n_out-1]:
                print( " %0.3f*%s"  % (item[1] , terms[item[0]]) , end=' ')
            print()
    else:
        m = max(svd.components_[topic])
        idx = {i:abs(j) for i, j in enumerate(svd.components_[topic])}
        sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
        weight = np.mean([ item[1] for item in sorted_idx[0:n_weight] ])
        print("* T %s) weight: %0.2f" % (topic, weight), end=' ')
        for item in sorted_idx[0:n_out-1]:
            print( " %0.3f*%s"  % (item[1] , terms[item[0]]) , end=' ')
        print()

def plot_clusters(svdX, y_pred, centers):
    plt.style.use('fivethirtyeight')
    f, ax1 = plt.subplots(1, 1, figsize=( 16, 8), facecolor='white')
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("K-Means")
    # Only plots the first 2 dimensions of the svdX matrix
    ax1.scatter(svdX[:,0], svdX[:,1], c=y_pred, cmap=plt.cm.Paired, s=45)
    ax1.scatter(centers[:, 0], centers[:, 1], marker='o', c="black", alpha=1, s=150)
    ax1.axis('off')
    plt.show()


# -------------------------------------
#  Params
# -------------------------------------

n_components    = 5     # Number of dimension for TruncatedSVD
n_clusters      = 2

db = connect('twitter')

# Get the already tokenized version of the timelines
documents = get_documents('alexip')

# This is hacky and due to the fact that we re-use previously tokenized documents
# We re assemble the tokens prior to tokenizing them again
tokenized   = [ ' '.join(doc['tokens']) for doc in documents ]

vectorizer  = TfidfVectorizer(max_df=0.9,  min_df=6,
                            max_features=500, use_idf=True,
                            strip_accents='ascii')

# X contains token frequency for each token
X = vectorizer.fit_transform(tokenized)

# SVD decomposition
svd  = TruncatedSVD(n_components, random_state= 10)
svdX = svd.fit_transform(X)

# Normalization.
# Note: for 2 dimensions this will cause the points to be on an ellipse.
# Comment the 2 lines below to produce more meaningful plots

nlzr = Normalizer(copy=False)
svdX = nlzr.fit_transform(svdX)

# Clustering
km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=4,
                verbose=False, random_state= 10)
km.fit(svdX)

print(" --------------------- ")
print("   Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(svdX, km.labels_, sample_size=1000))
print(" --------------------- ")

# Array mapping from words integer indices to actual words
terms   = vectorizer.get_feature_names()

display_topics(svd, terms, n_components)

# to plot the documents and clusters centers
# Only relevant for K = 2

y_pred  = km.predict(svdX)
centers = km.cluster_centers_

plot_clusters(svdX, y_pred, centers)
