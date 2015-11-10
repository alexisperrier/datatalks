import ast
import csv
import nltk
import re, sys, time, os, time
import random
import logging
import pandas as pd
import numpy as np
from string import digits
from time import time
from textblob import TextBlob
from textblob.wordnet import VERB
from textblob import Word

load_transcript = lambda x : ast.literal_eval(open(x, "r").read())

def by_speaker(transcript,speaker):
    return [c[speaker] for c in transcript if speaker in c.keys()]

filenames = {
    'dem_2015_1' : 'texts/dem_2015_1_list.txt',
    'gop_2015_1' : 'texts/gop_2015_1_list.txt',
    'gop_2015_2' : 'texts/gop_2015_2_list.txt',
    'gop_2015_3' : 'texts/gop_2015_3_list.txt',
}

csv_filenames = {
    'dem_2015_1' : 'texts/dem_2015_1_list.csv',
    'gop_2015_1' : 'texts/gop_2015_1_list.csv',
    'gop_2015_2' : 'texts/gop_2015_2_list.csv',
    'gop_2015_3' : 'texts/gop_2015_3_list.csv',
}

candidates = {
    'dem_2015_1' : ['CHAFEE', 'CLINTON', "O'MALLEY", 'SANDERS', 'WEBB'],
    'gop_2015_1' : ['BUSH','CARSON','CHRISTIE','CRUZ','HUCKABEE','KASICH','PAUL','RUBIO', 'TRUMP', 'WALKER'],
    'gop_2015_2' : ['BUSH','CARSON','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO', 'TRUMP', 'WALKER'],
    'gop_2015_3' : ['BUSH','CARSON','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO', 'TRUMP'],
}

candidates = {
    'dem_2015_1' : ['CLINTON','SANDERS'],
    'gop_2015_1' : ['BUSH','CARSON','TRUMP'],
    'gop_2015_2' : ['BUSH','CARSON','TRUMP'],
    'gop_2015_3' : ['BUSH','CARSON','TRUMP'],
}

tags = list(filenames.keys())

tag = 'gop_2015_1'

filename    = filenames[tag]
speakers    = candidates[tag]
transcript  = load_transcript(filename)
# n_split     = 5
number_of_words = 2000 # 2-3 pages long

for p in speakers:
    text = by_speaker(transcript,p)
    words = " ".join(text)
    print("%s: %s interventions, %s words %0.1f" % (p, len(text), len(words), len(words)/ len(text)   )  )

# splits the speaker intervention in n_split documents.
# this will multiply the number of documents and allow docs to be more subject
# specific
def split_text(text, n_split):
    res_text = []
    step = int(len(text) / n_split) + 1
    for i in range(n_split):
        m = min((i+1)*step -1, len(text) -1)
        res_text.append(  ' '.join(text[i*step: m ])  )
    return res_text

def chunk_text(number_of_words, whole_text):
    blob     = TextBlob(whole_text)
    n_split  = int(len(whole_text) / number_of_words)
    res_text = []
    i = 0
    for k in range(n_split):
        chunk = ""
        while len(chunk) < number_of_words and (i < len(blob.sentences)):
           chunk += ' ' + str(blob.sentences[i])
           i +=1
        res_text.append(chunk)
    # if there's a tail of sentences
    if i < len(blob.sentences):
        chunk = ""
        for j in range(i,len(blob.sentences)):
            chunk += ' ' + str(blob.sentences[i])
            i +=1
        res_text.append(chunk)
    return res_text

# --------------------------------------
#  Noun phrases
# --------------------------------------
def main_noun_phrases():
    for tag in tags:
        filename    = filenames[tag]
        speakers    = candidates[tag]
        transcript  = load_transcript(filename)

        party   = tag.split('_')[0]
        year    = int(tag.split('_')[1])
        debate  = int(tag.split('_')[2])
        np = []
        for speaker in speakers:
            text    = by_speaker(transcript,speaker)
            blob    = TextBlob(' '.join(text))
            [np.append(m) for m in  blob.noun_phrases if len(m.split()) >1]

# ends up with
noun_phrases = {'tax plan': 'taxplan', 'tax code': 'taxcode',
'american media': 'americanmedia', 'american people': 'americanpeople',
'american politics': 'americanpolitics', 'baby boomers': 'babyboomers',
'bank account': 'bankaccount', 'barack obama': 'obama', 'ben bernanke':
'benbernanke', 'bernie madolf': 'berniemadolf', 'bernie sanders': 'sanders',
'big data': 'bigdata', 'big government': 'biggovernment', 'business man': 'businessman',
'business owner': 'businessowner', 'carly fiorina': 'fiorina', 'tax credit': 'taxcredit',
'comic book': 'comicbook', 'comic-book': 'comicbook', 'conservative movement': 'conservativemovement',
'corporate taxes': 'corporatetaxes', 'corporate welfare': 'corporatewelfare',
'david petraeus': 'davidpetraeus', 'donald trump': 'trump',
'debt limit': 'debtlimit', 'education department': 'educationdepartment',
'fantasy football': 'fantasyfootball', 'foreign policy': 'foreignpolicy',
'gross domestic product': 'grossdomesticproduct', 'health care': 'healthcare',
'health insurance': 'healthinsurance', 'high school': 'highschool',
'hillary clinton': 'clinton', 'income tax': 'incometax',
'job market': 'jobmarket', 'john kerry': 'kerry', 'mccain': 'johnmccain',
'justice department': 'justicedepartment', 'law enforcement': 'lawenforcement',
'mainstream media': 'mainstreammedia', 'middle class': 'middleclass',
'mr. carson': 'carson', 'mrs. clinton': 'clinton', 'permanent residency': 'permanentresidents',
'permanent residents': 'permanentresidents', 'planned parenthood': 'plannedparenthood',
'public safety': 'publicsafety', 'rand paul': 'paul', 'road map': 'roadmap',
'sallie mae': 'salliemae', 'small business': 'smallbusiness', 'social security': 'socialsecurity',
'solar energy': 'solarenergy', 'tax code': 'taxcode', 'town hall': 'townhall',
'wall street': 'wallstreet', 'white house': 'whitehouse', 'silicon valley': 'siliconvalley',
'mike huckabee':'huckabee', 'ted cruz':'cruz','president obama':'obama',
'ben carson':'carson', 'chris christie':'christie','new jersey':'newjersey',
'united states':'unitedstates','mr. trump':'trump','u.s. troops':'ustroops',
'prime minister':'primeminister', 'mr. putin':'putin','vladimir putin':'putin',
'jeb bush':'jebbush'
}

# --------------------------------------
#  Export to csv file
# --------------------------------------
with open('texts/split_debates_2000.csv', 'w') as f:
    w = csv.DictWriter(f,['year','party','debate','candidate','text','text_noun'])
    w.writeheader()
    for tag in tags:

        filename    = filenames[tag]
        speakers    = candidates[tag]
        transcript  = load_transcript(filename)

        party   = tag.split('_')[0]
        year    = int(tag.split('_')[1])
        debate  = int(tag.split('_')[2])

        for speaker in speakers:
            whole_text      = ' '.join(by_speaker(transcript,speaker))
            text_chunked    = chunk_text(number_of_words, whole_text)
            print("file:%s speaker:%s %s chunks" % (tag, speaker, len(text_chunked)))
            for i in range(len(text_chunked)):
                chunk    = text_chunked[i]
                for o,n in noun_phrases.items():
                    chunk = chunk.lower()
                    chunk = chunk.replace(o,n)
                # blob    = TextBlob(chunk)
                # n       = [ word for word, pos in blob.tags if pos == 'NN' ]
                # v       = [ word for word, pos in blob.tags if pos == 'VBP' ]
                # nouns   = [ word for word in n if n not in v]
                # nouns   = ' '.join([ word for word, pos in blob.tags if pos == 'NN' ])
                # verbs   = ' '.join([ word for word, pos in blob.tags if pos == 'VBP' ])

                row = {
                    'year' : year,
                    'party': party,
                    'debate': debate,
                    'candidate': speaker,
                    'text': chunk,
                    # 'text_noun': nouns,
                }
                w.writerow(row)



with open('texts/debate_counts.csv', 'w') as f:
    w = csv.DictWriter(f,['Debate','candidate','interventions','words','n_interventions','sentiment'])
    w.writeheader()
    for tag in tags:
        filename    = filenames[tag]
        speakers    = candidates[tag]
        transcript  = load_transcript(filename)
        for speaker in speakers:
            text = by_speaker(transcript,speaker)
            blob = TextBlob(' '.join(text))
            print(blob.sentiment)
            debate  = tag.split('_')[0].upper() + ' ' + tag.split('_')[2]

            words = " ".join(text)
            row = {
                'Debate' : debate,
                'candidate': speaker,
                'interventions': len(text),
                'words': len(words),
                'n_interventions': len(text)/ len(transcript),
                'sentiment': blob.sentiment[0]
            }
            w.writerow(row)
