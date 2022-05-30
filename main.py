#!/usr/bin/env python
# coding: utf-8

# # Analyzing Twitter Users' Reflections using NLP

# This is a Sentiment Analysis Project using Natural Language Processing (NLP) Techniques.
# ## Contents
# 
# [1. Import Libraries](#1.-Import-Libraries)
# 
# [2. Tweets Mining](#2.-Tweets-Mining)
# 
# [3. Data Cleaning](#3.-Data-Cleaning) 
# 
# [4. Location Geocoding](#4.-Location-Geocoding)
# 
# [5. Tweets Processing](#5.-Tweets-Processing)
# 
# [6. Data Exploration](#6.-Data-Exploration)
# 
# [7. Sentiment Analysis](#7.-Sentiment-Analysis)
# 
# [8. Closing Remarks and Links](#THANK-YOU-FOR-TAKING-TIME-TO-STUDY-MY-NOTEBOOK.-I-HOPE-YOU-GOT-SOME-INSIGHTS.)

# ## 1. Import Libraries
# 


import tweepy # for tweet mining
import pandas as pd # for data manipulation and analysis
import numpy as np # for working with arrays and carrying out mathematical operations. Pandas is built on Numpy
import csv # to read and write csv files
import re # In-built regular expressions library
import string # Inbuilt string library
import glob # to retrieve files/pathnames matching a specified pattern. 
import random # generating random numbers
import requests # to send HTTP requests
from PIL import Image # for opening, manipulating, and saving many different image file f
import matplotlib.pyplot as plt # for plotting

# Set the limits for Pandas Dataframe display to avoid potential system freeze
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", 15)
pd.set_option('display.max_colwidth', 40)

# Natural Language Processing Toolkit
from nltk.corpus import stopwords, words # get stopwords from NLTK library & get all words in english language
from nltk.tokenize import word_tokenize # to create word tokens
# from nltk.stem import PorterStemmer (I played around with Stemmer and decided to use Lemmatizer instead)
from nltk.stem import WordNetLemmatizer # to reduce words to orginal form
from nltk import pos_tag # For Parts of Speech tagging

from textblob import TextBlob # TextBlob - Python library for processing textual data

import plotly.express as px # To make express plots in Plotly
import chart_studio.tools as cst # For exporting to Chart studio
import chart_studio.plotly as py # for exporting Plotly visualizations to Chart Studio
import plotly.offline as pyo # Set notebook mode to work in offline
pyo.init_notebook_mode()
import plotly.io as pio # Plotly renderer
import plotly.graph_objects as go # For plotting plotly graph objects
from plotly.subplots import make_subplots #to make more than one plot in Plotly


# WordCloud - Python linrary for creating image wordclouds
from wordcloud import WordCloud

from emot.emo_unicode import UNICODE_EMO, EMOTICONS # For emojis


# ## 2. Tweets Mining
# 



consumer_key = 'XXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_key= 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # Pass in Consumer key and secret for authentication by API
auth.set_access_token(access_key, access_secret) # Pass in Access key and secret for authentication by API
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True) # Sleeps when API limit is reached


# ### Creating User-defined Functions for Tweets Mining
# I created 4 different functions for the different search queries I had and stored them in a csv file. This is because I ran the program for 3 consecutive days 22nd to 25th of December. 
# For the 1st run, I did not have to specify the __"since_id"__ but I had to for the following days so that Twitter API does not return Tweets I already had. 
# Another thing to note is you don't have to specify a sleep time for your function. Tweepy has a built-in attribute "wait_on_rate_limit" which I specified above.


def get_tweets2(search_query2, num_tweets2, since_id_num2):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list2 = [tweets for tweets in tweepy.Cursor(api.search,
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
        with open('tweets_2020_has_been.csv','a', newline='', encoding='utf-8') as csvFile2:
            csv_writer2 = csv.writer(csvFile2, delimiter=',') # create an instance of csv object
            csv_writer2.writerow([tweet_id, created_at, text, location, retweet, favorite]) # write each row



search_words2 = "\"2020 has been\"" # Specifying exact phrase to search
# Exclude Links, retweets, replies
search_query2 = search_words2 + " -filter:links AND -filter:retweets AND -filter:replies" 
with open('tweets_2020_has_been.csv', encoding='utf-8') as data:
    latest_tweet = int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
get_tweets2(search_query2, 10000, latest_tweet)



# Same as above
def get_tweets3(search_query3, num_tweets3, since_id_num3):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list3 = [tweets for tweets in tweepy.Cursor(api.search,
                                    q=search_query3,
                                    since_id=since_id_num3,
                                    lang="en",
                                    tweet_mode='extended').items(num_tweets3)]

    # Begin scraping the tweets individually:
    for tweet in tweet_list3[::-1]:
        tweet_id = tweet.id
        created_at = tweet.created_at
        text = tweet.full_text
        location = tweet.user.location
        retweet = tweet.retweet_count
        favorite = tweet.favorite_count
        with open('tweets_2020_was_a.csv', 'a', newline='', encoding='utf-8') as csvFile3:
            csv_writer3 = csv.writer(csvFile3, delimiter=',')
            csv_writer3.writerow([tweet_id, created_at, text, location, retweet, favorite])



search_words3 = "\"2020 was a\"" # Specifying exact phrase to search
# Exclude Links, retweets, replies
search_query3 = search_words3 + " -filter:links AND -filter:retweets AND -filter:replies"
with open('tweets_2020_was_a.csv', encoding='utf-8') as data:
    latest_tweet = int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
get_tweets3(search_query3, 10000, latest_tweet)




# Same as above
def get_tweets4(search_query4, num_tweets4, since_id_num4):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list4 = [tweets for tweets in tweepy.Cursor(api.search,
                                    q=search_query4,
                                    lang="en",
                                    since_id=since_id_num4,
                                    tweet_mode='extended').items(num_tweets4)]

    # Begin scraping the tweets individually:
    for tweet in tweet_list4[::-1]:
        tweet_id = tweet.id
        created_at = tweet.created_at
        text = tweet.full_text
        location = tweet.user.location
        retweet = tweet.retweet_count
        favorite = tweet.favorite_count
        with open('tweets_this_year_has_been.csv','a', newline='', encoding='utf-8') as csvFile4:
            csv_writer4 = csv.writer(csvFile4, delimiter=',')
            csv_writer4.writerow([tweet_id, created_at, text, location, retweet, favorite])



search_words4 = "\"this year has been\"" # Specifying exact phrase to search
# Exclude Links, retweets, replies
search_query4 = search_words4 + " -filter:links AND -filter:retweets AND -filter:replies"
with open('tweets_this_year_has_been.csv',encoding='utf-8') as data:
    latest_tweet=int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
get_tweets4(search_query4,10000,latest_tweet)



def get_tweets5(search_query5, num_tweets5, since_id_num5):
    # Collect tweets using the Cursor object
    # Each item in the iterator has various attributes that you can access to get information about each tweet
    tweet_list5 = [tweets for tweets in tweepy.Cursor(api.search,
                                    q=search_query5,
                                    lang="en",
                                    since_id=since_id_num5,
                                    tweet_mode='extended').items(num_tweets5)]

    # Begin scraping the tweets individually:
    for tweet in tweet_list5[::-1]:
        tweet_id = tweet.id
        created_at = tweet.created_at
        text = tweet.full_text
        location = tweet.user.location
        retweet = tweet.retweet_count
        favorite = tweet.favorite_count
        with open('tweets_this_year_was_a.csv','a',newline='', encoding='utf-8') as csvFile5:
            csv_writer5 = csv.writer(csvFile5, delimiter=',')
            csv_writer5.writerow([tweet_id, created_at, text, location, retweet, favorite])




search_words5 = "\"this year was a\"" # Specifying exact phrase to search
search_query5 = search_words5 + " -filter:links AND -filter:retweets AND -filter:replies"
with open('tweets_this_year_was_a.csv', encoding='utf-8') as data:
    latest_tweet = int(list(csv.reader(data))[-1][0]) # Return the most recent tweet ID
get_tweets5(search_query5, 10000, latest_tweet)


# ### Combining all Tweets into single Pandas Dataframe


path = r'./Data'  # use your path
all_files = glob.glob(path + "/*.csv")

tweets = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0) # Convert each csv to a dataframe
    tweets.append(df)

tweets_df = pd.concat(tweets, axis=0, ignore_index=True) # Merge all dataframes

tweets_df.head()


# ## 3. Data Cleaning
# The dataframe in Section 2 were cleaned in this section. Duplicate values were checked and removed. It is also important to mention that the Tweet ID was considered as the Primary key for all the dataframe. 
# I also replaced "NaN" values in Location column because if used for Location Geocoding, "NaN" values return Coordinates which should not be.


tweets_df.shape #Get number of rows and columns

tweets_df.duplicated(subset='tweet_id').sum() # Check for duplicate values




tweets_df=tweets_df.drop_duplicates(subset=['tweet_id']) # drop duplicate values




tweets_df.shape # Check the shape after dropping duplicates




tweets_df.isna().any() # Check for "NaN" values




tweets_df['location']=tweets_df['location'].fillna('No location') # Replace "NaN" values with "No Location"

tweets_df.isna().any() # Check for "NaN" values again


# ## 4. Location Geocoding
# For my final dashboard, I wanted to add a map that shows the number of tweets per country. To do that, Tableau needs basic geographic information that it can recognize. After several trials and errors, 
# I ended up using the HERE Developer API to return Longitude, Latitude & Country names for each tweet. One key thing is if you send a request with a "NaN" or Null value to the API, it will return an actual location.
#  This is why I had to replace "NaN" values with "No Location" in the step on Data Cleaning. __It is also important to study your dataframe to ensure you are getting the expected results. This is how I caught this discrepancy__



URL = "https://geocode.search.hereapi.com/v1/geocode"  # Deevloper Here API link
api_key = '#######################'  # Acquire api key from developer.here.com

def getCoordinates(location):
    PARAMS = {'apikey': api_key, 'q': location} # required parameters
    r = requests.get(url=URL, params=PARAMS)  # pass in required parameters
    # get raw json file. I did  this because when I combined this step with my "getLocation" function, 
    # it gave me error for countries with no country_code or country_name. Hence, I needed to use try - except block
    data = r.json() # Raw json file 
    return data


tweets_df['Location_data']=tweets_df['location'].apply(getCoordinates) # Apply getCoordinates Function


# a function to extract required coordinates information to the tweets_df dataframe

def getLocation(location):
    for data in location:
        if len(location['items'])>0:
            latitude = location['items'][0]['position']['lat']
            longitude = location['items'][0]['position']['lng']
            try:   
                country_code = location['items'][0]['address']['countryCode']
                country_name = location['items'][0]['address']['countryName']
                country_name = location['items'][0]['address']['countryName']
            except KeyError:
                country_code = float('Nan')
                country_name = float('Nan')
        else: 
            latitude = float('Nan')
            longitude = float('Nan')
            country_code = float('Nan') 
            country_name = float('Nan')
        result = (latitude, longitude, country_code, country_name)
    return result



tweets_df['location']=tweets_df['location_data'].apply(getLocation) #apply getLocation function


tweets_df.head() # Check dataframe first 5 rows

# Extraction of Location Coordinates and Country names to different columns
tweets_df[['Latitude', 'Longitude', 'Country_Code',
           'Country_Name']] = pd.DataFrame(tweets_df['Location'].tolist(),
                                           index=tweets_df.index)


tweets_df.head() # Check dataframe first 5 rows




# Drop unnecessary columns
tweets_df.drop(['Location_data','location','Location'], axis=1, inplace=True)
# Rename columns
tweets_df.columns=['Tweet_ID','Time_Created','Tweet','Retweet_Count','Favorite_Count',
                   'Latitude','Longitude','Country_Code','Country_Name']



tweets_df.head() # Check dataframe first 5 rows


# ## 5. Tweets Processing
# To reach the ultimate goal, there was a need to clean up the individual tweets. To make this easy, I created several functions in my Python code which I further applied to the "Tweets" column in my Pandas data frame to produce the desired results. 
# Also, for my Word Cloud, I wanted to only show the words used to describe 2020, so I created a function to extract only the adjectives to a new column.


# Function to remove punctuations, links, emojis, and stop words
def preprocessTweets(tweet):
    tweet = tweet.lower()  #has to be in place
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#|\d+', '', tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)  # convert string to tokens
    filtered_words = [w for w in tweet_tokens if w not in stop_words]
    filtered_words = [w for w in filtered_words if w not in emojis]
    filtered_words = [w for w in filtered_words if w in word_list]

    # Remove punctuations
    unpunctuated_words = [char for char in filtered_words if char not in string.punctuation]
    unpunctuated_words = ' '.join(unpunctuated_words)

    return "".join(unpunctuated_words)  # join words with a space in between them


# function to obtain adjectives from tweets
def getAdjectives(tweet):
    tweet = word_tokenize(tweet)  # convert string to tokens
    tweet = [word for (word, tag) in pos_tag(tweet)
             if tag == "JJ"]  # pos_tag module in NLTK library
    return " ".join(tweet)  # join words with a space in between them



# Defining my NLTK stop words and my user-defined stop words
stop_words = list(stopwords.words('english'))
user_stop_words = ['2020', 'year', 'many', 'much', 'amp', 'next', 'cant', 'wont', 'hadnt',
                    'havent', 'hasnt', 'isnt', 'shouldnt', 'couldnt', 'wasnt', 'werent',
                    'mustnt', '’', '...', '..', '.', '.....', '....', 'been…', 'one', 'two',
                    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'aht',
                    've', 'next']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + user_stop_words + alphabets
word_list = words.words()  # all words in English language
emojis = list(UNICODE_EMO.keys())  # full list of emojis


# ### An example of how the above function works is shown below


print(preprocessTweets("2020 was a year of difficulty. It was sad, but I am happy with how things worked out for me"))


# Apply preProcessTweets function to the 'Tweet' column to generate a new column called 'Processed Tweets'.
# This took 23 mins to run for 50,780 rows
tweets_df['Processed_Tweets'] = tweets_df['Tweet'].apply(preprocessTweets)


# Apply getAdjectives function to the new 'Processed Tweets' column to generate a new column called 'Tweets_Adjectives'
tweets_df['Tweets_Adjectives'] = tweets_df['Processed_Tweets'].apply(getAdjectives)


tweets_df.head() # Check dataframe first 5 rows



# function to return words to their base form using Lemmatizer
def preprocessTweetsSentiments(tweet):
    tweet_tokens = word_tokenize(tweet)
    lemmatizer = WordNetLemmatizer() # instatiate an object WordNetLemmatizer Class
    lemma_words = [lemmatizer.lemmatize(w) for w in tweet_tokens]
    return " ".join(lemma_words)




# Apply preprocessTweetsSentiments function to the 'Processed Tweets' column to generate a new column
# called 'Processed_Tweets'
tweets_df['Tweets_Sentiments'] = tweets_df['Processed_Tweets'].apply(preprocessTweetsSentiments)


tweets_df.head() # Check dataframe first 5 rows


# I had to write my results to a csv file at every instance due to the amount of time it took for the preprocessTweets
#  function to run
tweets_df.to_csv('Tweets_Processed.csv',encoding='utf-8-sig', index=False) 
# Also, encoding is important when writing text to csv file


# ## 6. Data Exploration
# In this section, the aim was to show the most common words used by Twitter Users to describe 2020. This was made possible by the "getAdjectives" function in Section 5. I also made use of WordCloud and MatPlotlib for this task.


# Extract all tweets into one long string with each word separate with a "space"
tweets_long_string = tweets_df['Tweets_Adjectives'].tolist()
tweets_long_string = " ".join(tweets_long_string)


# Import Twitter Logo
image = np.array(Image.open('twitter.png'))
    
fig = plt.figure() # Instantiate the figure object
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height

plt.imshow(image, cmap=plt.cm.gray, interpolation='bilinear') # Display data as an image
plt.axis('off') # Remove axis
plt.show() # Display image



# Create function to generate the blue colour for the Word CLoud

def blue_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(210, 100%%, %d%%)" % random.randint(50, 70)



# Instantiate the Twitter word cloud object
twitter_wc = WordCloud(background_color='white', max_words=1500, mask=image)

# generate the word cloud
twitter_wc.generate(tweets_long_string)

# display the word cloud
fig = plt.figure()
fig.set_figwidth(14)  # set width
fig.set_figheight(18)  # set height

plt.imshow(twitter_wc.recolor(color_func=blue_color_func, random_state=3),
           interpolation="bilinear")
plt.axis('off')
plt.show()


twitter_wc.to_file("wordcloud.png") #save to a png file


# #### Analyzing Top Words in the Word Cloud


# Combine all words into a list
tweets_long_string = tweets_df['Tweets_Adjectives'].tolist()
tweets_list=[]
for item in tweets_long_string:
    item = item.split()
    for i in item:
        tweets_list.append(i)



# Use the Built-in Python Collections module to determine Word frequency
from collections import Counter
counts = Counter(tweets_list)
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df.columns = ['Words', 'Count']
df.sort_values(by='Count', ascending=False, inplace=True)



df.head(10)  # Check dataframe first 10 rows


# ### Top 10 Words in Twitter Users' 2020 Reflections


# print(px.colors.sequential.Blues_r) to get the colour list used here. Please note, I swatched some colours

# Define my colours for the Plotly Plot
colors = ['rgb(8,48,107)', 'rgb(8,81,156)', 'rgb(33,113,181)', 'rgb(66,146,198)',
            'rgb(107,174,214)', 'rgb(158,202,225)', 'rgb(198,219,239)',
            'rgb(222,235,247)', 'rgb(247,251,255)', 'rgb(247,253,255)']

# Set layout for Plotly Subplots
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, { "type": "domain"}]],
                    vertical_spacing=0.001)

# Add First Plot
fig.add_trace(go.Bar(x = df['Count'].head(10), y=df['Words'].head(10),marker=dict(color='rgba(66,146,198, 1)',
            line=dict(color='Black'),),name='Bar Chart',orientation='h'), 1, 1)

# Add Second Plot
fig.add_trace(go.Pie(labels=df['Words'].head(10),values=df['Count'].head(15),textinfo='label+percent',
                    insidetextorientation='radial', marker=dict(colors=colors, line=dict(color='DarkSlateGrey')),
                    name='Pie Chart'), 1, 2)
# customize layout
fig.update_layout(shapes=[dict(type="line",xref="paper", yref="paper", x0=0.5, y0=0, x1=0.5, y1=1.0,
         line_color='DarkSlateGrey', line_width=1)])

# customize plot title
fig.update_layout(showlegend=False, title=dict(text="Twitter Users' 2020 Refelections <i>(10 Most Common Words)</i>",
                  font=dict(size=18, )))

# Customize backgroound, margins, axis, title
fig.update_layout(yaxis=dict(showgrid=False,
                             showline=False,
                             showticklabels=True,
                             domain=[0, 1],
                             categoryorder='total ascending',
                             title=dict(text='Common Words', font_size=14)),
                             xaxis=dict(zeroline=False,
                             showline=False,
                             showticklabels=True,
                             showgrid=True,
                             domain=[0, 0.42],
                             title=dict(text='Word Count', font_size=14)),
                             margin=dict(l=100, r=20, t=70, b=70),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)')

# Specify X and Y values for Annotations
x = df['Count'].head(10).to_list()
y = df['Words'].head(10).to_list()

# Show annotations on plot
annotations = [dict(xref='x1', yref='y1', x=xa + 350, y=ya, text=str(xa), showarrow=False) for xa, ya in zip(x, y)]

fig.update_layout(annotations=annotations)
fig.show(renderer = 'png')


# Export to Plot to Chart Studio using my Chart Studio Credentials
py.plot(fig, filename = 'Twitter Users 2020 Refelections (10 Most Common Words)', auto_open=True)


# ## 7. Sentiment Analysis
# In this section, the aim was to show the most common words used by Twitter Users to describe 2020. This was made possible byt the getAdjectives function. 
# I also made use of WordCloud and MatPlotlib for this task.


# Create function to obtain Subjectivity Score
def getSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

# Create function to obtain Polarity Score
def getPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

# Create function to obtain Sentiment category
def getSentimentTextBlob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"



# Apply all functions above to respective columns
tweets_df['Subjectivity']=tweets_df['Tweets_Sentiments'].apply(getSubjectivity)
tweets_df['Polarity']=tweets_df['Tweets_Sentiments'].apply(getPolarity)
tweets_df['Sentiment']=tweets_df['Polarity'].apply(getSentimentTextBlob)



# See quick results of the Sentiment Analysis
tweets_df['Sentiment'].value_counts()



# Create dataframe for Count of Sentiment Categories
bar_chart = tweets_df['Sentiment'].value_counts().rename_axis('Sentiment').to_frame('Total Tweets').reset_index()


bar_chart # Display dataframe



sentiments_barchart = px.bar(bar_chart, x = 'Sentiment', y='Total Tweets', color='Sentiment')

sentiments_barchart.update_layout(title='Distribution of Sentiments Results',
                                  margin={"r": 0, "t": 30, "l": 0, "b": 0})

sentiments_barchart.show(renderer = 'png') #Display plot. 

# I used renderer so that the plots show on the web when people view my notebook. 
# For interactive plots, exclude the "renderer" argument
# Note, I further customized the plot on Chart Studio for my Medium post




# Export to Plot to Chart Studio using my Chart Studio Credentials. 
py.plot(sentiments_barchart, filename = 'Distribution of Sentiments Results', auto_open=True)


# ## WE MADE IT!! 
# ### Preview the resulting dataframe in preparation for export to Tableau



tweets_df.head() # Check dataframe first 5 rows




# Remove Unnecessary columns. I used copy here because I wanted a datframe that does not impact my original dataframe
tableau_df = tweets_df.drop((['Processed_Tweets','Tweets_Sentiments','Subjectivity','Polarity']), axis=1).copy(deep=True)




tableau_df.head() # Check dataframe first 5 rows



# Export to Excel file. You can link Python to Tableau by the way but I Ihaven't learned that yet
tableau_df.to_excel('Tweets_Tableau_Final_File.xlsx', encoding='utf-8-sig', index=False)


# ## THANK YOU FOR TAKING TIME TO STUDY MY NOTEBOOK. I HOPE YOU GOT SOME INSIGHTS.
# ## You can use the links below to view my other pages
# 
# [Medium Page](https://jess-analytics.medium.com/)
# 
# [Tableau Dashboard](https://public.tableau.com/views/TwitterUsers2020ReflectionsDashboard/FinalDashboard?:language=en-GB&:display_count=y&publish=yes&:origin=viz_share_link)
# 
# [LinkedIn Page](https://www.linkedin.com/in/jessicauwoghiren/)
# 
# [Twitter Page](https://twitter.com/jessica_xls)
# 
# [My Data Community Placeholder Website](https://linktr.ee/DataTechSpace)
# 
# 
