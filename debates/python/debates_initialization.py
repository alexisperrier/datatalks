from __future__ import print_function
import ast
import csv
import nltk
import re, sys, time, os, time
import random
import logging
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from string import digits
from time import time
from collections import defaultdict

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
    'gop_2015_1' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','HUCKABEE','KASICH','PAUL','RUBIO', 'WALKER'],
    'gop_2015_2' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO', 'WALKER'],
    'gop_2015_3' : ['BUSH','CARSON','TRUMP','CHRISTIE','CRUZ','FIORINA','HUCKABEE','KASICH','PAUL','RUBIO'],
}

def by_speaker(transcript,speaker):
    return [c[speaker] for c in transcript if speaker in c.keys()]

load_transcript = lambda x : ast.literal_eval(open(x, "r").read())
list_speakers = lambda x : set([list(t.keys())[0] for t in x ])
