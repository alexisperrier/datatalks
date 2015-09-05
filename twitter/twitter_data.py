'''
This script retrieves the tweets of all followers of a given twitter account.
The data is stored in a MongoDB store.

The twitter and MongoDB credentials are stored in a cfg file
'''
# todo get more than 200 tweets per user

# todo: function to load twitter creds => returns twitter object
# todo: same for MongoDB
# get followers as a function

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

# twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
# twitter.verify_credentials()
twitter     = Twython(APP_KEY, APP_SECRET, oauth_version=2)
# ACCESS_TOKEN = twitter.obtain_access_token()
twitter     = Twython(APP_KEY, access_token=ACCESS_TOKEN)

# ---------------------------------------------------------
#  MongoDB connection
# ---------------------------------------------------------
DBNAME      = config['database']['name']
client      = MongoClient()
db          = client[DBNAME]

screen_name  = 'alexip'
n_max_folwrs = 700

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

tl= dict()
for id in flw_ids:
    try:
        # only retrieve tweets for user if we don't have them in store already
        twt = db.tweets.find({'user_id':id})
        if twt.count() == 0:
            handle_rate_limiting()
            params = {'user_id': id, 'count': 200, 'trim_user': 'true' }
            tl[id] = twitter.get_user_timeline(**params)

            # aggregate tweets
            text = ''
            for tw in tl[id]:
                text += ' ' + tw['text']

            # store document
            item = {
                'raw_text': text,
                'user_id': id,
            }
            tweets.insert_one(item)
    except TwythonRateLimitError as e:
        # Let's wait if we hit the Rate limit
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
documents    = [tw['raw_text']  for tw in db.tweets.find()]
print("We have " + str(len(documents)) + " documents ")
