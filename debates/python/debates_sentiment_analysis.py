# Sentiment analysis on 2015 presidential candidate debates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
from textblob.wordnet import VERB
from textblob import Word

filenames = {
    'dem_2015_1' : 'texts/dem_2015_1_list.txt',
    'gop_2015_1' : 'texts/gop_2015_1_list.txt',
    'gop_2015_2' : 'texts/gop_2015_2_list.txt',
    'gop_2015_3' : 'texts/gop_2015_3_list.txt',
}

candidates = {
    'dem_2015_1' : ['CHAFEE', 'CLINTON', "O'MALLEY", 'SANDERS', 'WEBB'],
    'gop_2015_1' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','HUCKABEE','KASICH','PAUL','RUBIO', 'WALKER'],
    'gop_2015_2' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO', 'WALKER'],
    'gop_2015_3' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO'],
}

tags = list(filenames.keys())

def by_speaker(transcript,speaker):
    return [c[speaker] for c in transcript if speaker in c.keys()]

load_transcript = lambda x : ast.literal_eval(open(x, "r").read())

# ------------------------------------------------------------------------
#  SA per candidate and debates
# ------------------------------------------------------------------------

df = pd.read_csv('texts/debates.csv')

with open('texts/debates_sentiment.csv', 'w') as f:
    w = csv.DictWriter(f,['debate','candidate','sentiment'])
    w.writeheader()
    for tag in tags:

        filename    = filenames[tag]
        speakers    = candidates[tag]
        transcript  = load_transcript(filename)

        for speaker in speakers:

            blob = TextBlob(' '.join(by_speaker(transcript,speaker)))
            row = {
                'debate': tag,
                'candidate': speaker,
                'sentiment': blob.sentiment[0],
            }
            w.writerow(row)


# ------------------------------------------------------------------------
#  Topic polarity
# ------------------------------------------------------------------------


# Extract all sentences containing a given word
def sentiment_in_sentences(word,blob):
    sentences = [sent for sent in blob.sentences if word in sent]

    score = 0
    for sent in sentences:
        score += sent.sentiment.polarity
        # print("[%s] %s" % (sent.sentiment.polarity, sent) )
    score = score / len(sentences)
    print("%s sentences: Score on %s: %0.2f" % (len(sentences),word,score))

# For Democrats
tag = 'dem_2015_1'
filename    = filenames[tag]
speakers    = candidates[tag]
transcript  = load_transcript(filename)

for p in speakers:
    blob = TextBlob(' '.join( by_speaker(transcript,p) ))
    sentiment_in_sentences('Obama',blob)

# Overall sentiment independent of candidates
all_text = [list(t.values())[0] for t in transcript]
sentiment_in_sentences('Obama', TextBlob(' '.join(all_text)) )

# and for Republicans
tag = 'gop_2015_3'
filename    = filenames[tag]
speakers    = candidates[tag]
transcript  = load_transcript(filename)

# Overall sentiment independent of candidates
all_text = [list(t.values())[0] for t in transcript]
sentiment_in_sentences('Obama', TextBlob(' '.join(all_text)) )
