import tweepy # for tweet mining
import pandas as pd # for data manipulation and analysis
import numpy as np # for working with arrays and carrying out mathematical operations. Pandas is built on Numpy
import csv # to read and write csv files



consumer_key = 'XXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'
access_key= 'XXXXXXXXXXXXXXXXXXXXXX'
access_secret = 'XXXXXXXXXXXXXXXXXXXXXXX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # Pass in Consumer key and secret for authentication by API
auth.set_access_token(access_key, access_secret) # Pass in Access key and secret for authentication by API
api = tweepy.API(auth,wait_on_rate_limit=True) # Sleeps when API limit is reached


def get_tweets1(search_query1, num_tweets1, filename):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list1 = [tweets for tweets in tweepy.Cursor(api.search_tweets,
                                    q=search_query1,
                                    lang="en",
                                    tweet_mode='extended').items(num_tweets1)]
    # Begin scraping the tweets individually:
    for tweet in tweet_list1[::-1]:
        tweet_id = tweet.id # get Tweet ID result
        created_at = tweet.created_at # get time tweet was created
        text = tweet.full_text # retrieve full tweet text
        location = tweet.user.location # retrieve user location
        retweet = tweet.retweet_count # retrieve number of retweets
        favorite = tweet.favorite_count # retrieve number of likes
        with open(filename,'a', newline='', encoding='utf-8') as csvFile1:
            csv_writer1 = csv.writer(csvFile1, delimiter=',') # create an instance of csv object
            csv_writer1.writerow([tweet_id, created_at, text, location, retweet, favorite]) # write each row


search_words1 = "\"IPL 2022 has \"" # Specifying exact phrase to search
# Exclude Links, retweets, replies
search_query1 = search_words1 + " -filter:links AND -filter:retweets AND -filter:replies" 
filename = 'Data\\IPL_2022_has.csv'
get_tweets1(search_query1, 5000, filename)

search_words2 = "\"IPL 2022 was \"" # Specifying exact phrase to search
search_query2 = search_words2 + " -filter:links AND -filter:retweets AND -filter:replies" 
# with open('Data\\IPL_2022_was_a.csv', encoding='utf-8') as data:
#     latest_tweet = int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
filename = 'Data\\IPL_2022_was.csv'
get_tweets1(search_query2, 5000, filename)


search_words3 = "\"IPL 2022 \"" # Specifying exact phrase to search
search_query3 = search_words3 + " -filter:links AND -filter:retweets AND -filter:replies" 
filename = 'Data\\IPL_2022.csv'
get_tweets1(search_query3, 5000, filename)


search_words4 = "\"This year IPL \"" # Specifying exact phrase to search
search_query4 = search_words4 + " -filter:links AND -filter:retweets AND -filter:replies" 
filename = 'Data\\This_year_IPL.csv'
get_tweets1(search_query4, 5000, filename)

def get_tweets2(search_query2, num_tweets2, since_id_num2):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list2 = [tweets for tweets in tweepy.Cursor(api.search_tweets,
                                    q=search_query2,
                                    lang="en",
                                    since_id=since_id_num2,
                                    tweet_mode='extended').items(num_tweets2)]
    # Begin scraping the tweets individually:
    for tweet in tweet_list2[::-1]:
        tweet_id = tweet.id # get Tweet ID result
        created_at = tweet.created_at # get time tweet was created
        text = tweet.full_text # retrieve full tweet text
        location = tweet.user.location # retrieve user location
        retweet = tweet.retweet_count # retrieve number of retweets
        favorite = tweet.favorite_count # retrieve number of likes
        with open('Data\\IPL_2022_was_a.csv','a', newline='', encoding='utf-8') as csvFile2:
            csv_writer2 = csv.writer(csvFile2, delimiter=',') # create an instance of csv object
            csv_writer2.writerow([tweet_id, created_at, text, location, retweet, favorite]) # write each row


# search_words2 = "\"IPL 2022 was a\"" # Specifying exact phrase to search
# search_query2 = search_words2 + " -filter:links AND -filter:retweets AND -filter:replies" 
# with open('Data\\IPL_2022_has_been.csv', encoding='utf-8') as data:
#     latest_tweet = int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
# get_tweets2(search_query2, 5000, latest_tweet)