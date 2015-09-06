'''
This script retrieves the tweets of all followers of a given twitter account.
The data is stored in a MongoDB store.

The twitter and MongoDB credentials are stored in a cfg file

Python: 3.4
'''

import sys, time
from configparser import ConfigParser
from pymongo import MongoClient
from twython import Twython, TwythonRateLimitError

# This function allows us to check the remaining statuses and applications
# limits imposed by twitter.
# When the app_status or the timeline_status is exhausted, forces a wait
# for the period indicated by twitter
def handle_rate_limiting():
    # prepopulating this to make the first 'if' check fail
    app_status = {'remaining':1}
    while True:
        if app_status['remaining'] > 0:
            status = twitter.get_application_rate_limit_status(resources =
                                                    ['statuses', 'application'])
            status = status['resources']
            app_status = status['application']['/application/rate_limit_status']
            timeline_status = status['statuses']['/statuses/user_timeline']
            if timeline_status['remaining'] == 0:
                wait = max(timeline_status['reset'] - time.time(), 0) + 1
                time.sleep(wait)
            else:
                return
        else:
            wait = max(app_status['reset'] - time.time(), 0) + 10
            time.sleep(wait)

# ---------------------------------------------------------
#  Twitter Connection
# ---------------------------------------------------------

config = ConfigParser()
config.read('twitter.cfg')
APP_KEY      = config['credentials']['app_key']
APP_SECRET   = config['credentials']['app_secret']
ACCESS_TOKEN = config['credentials']['access_token_oauth_2']

twitter     = Twython(APP_KEY, APP_SECRET, oauth_version=2)
twitter     = Twython(APP_KEY, access_token=ACCESS_TOKEN)

# ---------------------------------------------------------
#  MongoDB connection
# ---------------------------------------------------------
DBNAME      = config['database']['name']
client      = MongoClient()
db          = client[DBNAME]

screen_name  = 'alexip'     # The main twitter account
n_max_folwrs = 700          # The number of followers to consider

# ---------------------------------------------------------
#  1) get follower ids
#  see https://dev.twitter.com/rest/reference/get/followers/ids
# ---------------------------------------------------------
flwrs   = twitter.get_followers_ids(screen_name = screen_name,
                                    count = n_max_folwrs)
flw_ids = flwrs['ids']
flw_ids.sort()

# insert follower_ids in db
db.followers.insert_one({"follower_ids": flw_ids, "user": "alexip"})

# ---------------------------------------------------------
#  Get 200 tweets per follower
#  (200 is the maximum number of tweets imposed by twitter)
# ---------------------------------------------------------
for id in flw_ids:
    try:
        # only retrieve tweets for user if we don't have them in store already
        twt = db.tweets.find({'user_id':id})
        handle_rate_limiting()
        params = {'user_id': id, 'count': 200, 'contributor_details': 'true' }
        tl = twitter.get_user_timeline(**params)
        # aggregate tweets
        text = ' '.join( [tw['text'] for tw in tl])

        item = {
            'raw_text': text,
            'user_id': id,
            'n_tweets': len(tl),
            'screen_name': tl[0]['user']['screen_name'],
            'lang': tl[0]['lang'],
        }

        if twt.count() == 0:
            # store document
            tweets.insert_one(item)
        else:
            # update the record
            res = db.tweets.replace_one( {'user_id':id}, item )
            if res.matched_count == 0:
                print("no match for id: ",id)
            else:
                if res.modified_count == 0:
                    print("no modification for id: ",id)
                else:
                    print("replaced id ",tl[0]['user']['screen_name'],
                            id,len(tl), tl[0]['lang'] )
    except TwythonRateLimitError as e:
        # Wait if we hit the Rate limit
        reset = int(twitter.get_lastfunction_header('x-rate-limit-reset'))
        wait = max(reset - time.time(), 0) + 10 # addding 10 second pad
        print("[Exception Raised] Rate limit exceeded waiting: %s", wait)
        time.sleep(wait)
    except:
        print(" FAILED:", id)
        print("Unexpected error:", sys.exc_info()[0])
        pass

# ---------------------------------------------------------
#  check how many documents we now have in the Database
# ---------------------------------------------------------
follower_docs = db.tweets.find()
documents    = [tw['raw_text']  for tw in follower_docs]
print("We have " + str(len(documents)) + " documents ")

n_tweets = sum([tw['n_tweets']  for tw in follower_docs if 'n_tweets' in tw.keys()])
print("Total number of tweets: ", n_tweets)
print("On average #tweets per document: ", n_tweets / len(documents))
