# Word cloud generation for the 2015 presidential debates
# see http://sebastianraschka.com/Articles/2014_twitter_wordcloud.html
# pip install git+git://github.com/amueller/word_cloud.git

import pandas as pd
from wordcloud import WordCloud
from textblob import TextBlob
from textblob import Word


def generate_wordcloud(speakers, df):
    '''
     Generates wordcloud files in the /data folder
     one for each name in the speakers list
     the df dataframe has the following columns:
        'year', 'party', 'debate', 'candidate', 'text'
    '''
    for p in speakers:
        print("Generating: %s" % p)
        wordcloud_filename = 'data/wordcloud_' + p.lower() + '.png'
        wordcloud = WordCloud( max_words=500, width=900, height=600, background_color='white')
        text    =  df[df.candidate == p]['text'].get_values()[0]
        blob    = TextBlob(text)
        nouns   = [ word for word, pos in blob.tags if pos == 'NN' ]

        wordcloud.generate(' '.join(nouns))
        wordcloud.to_file(wordcloud_filename)

df = pd.read_csv('../texts/debates.csv')

# ---------------------------------------------------------
#  Last GOP debate for Trump, Buch and Carson
# ---------------------------------------------------------

speakers    = ['BUSH', 'CARSON', 'TRUMP']
df_gop3     = df[df.debate == 3]
generate_wordcloud(speakers, df_gop3)

# ---------------------------------------------------------
#  First Democrat debate for Clinton and Sanders
# ---------------------------------------------------------
speakers    = ['CLINTON', 'SANDERS',"O'MALLEY"]
df_dem1     = df[df.party == 'dem']
generate_wordcloud(speakers, df_dem1)

