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
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
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
#pyo.init_notebook_mode()
import plotly.io as pio # Plotly renderer
import plotly.graph_objects as go # For plotting plotly graph objects
from plotly.subplots import make_subplots #to make more than one plot in Plotly


# WordCloud - Python linrary for creating image wordclouds
from wordcloud import WordCloud

from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO # For emojis


path = r'./Data'  # use your path
all_files = glob.glob(path + "/*.csv")

tweets = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0) # Convert each csv to a dataframe
    tweets.append(df)

tweets_df = pd.concat(tweets, axis=0, ignore_index=True) # Merge all dataframes

print(tweets_df.head())


# ## 3. Data Cleaning
# The dataframe in Section 2 were cleaned in this section. Duplicate values were checked and removed. It is also important to mention that the Tweet ID was considered as the Primary key for all the dataframe. 
# I also replaced "NaN" values in Location column because if used for Location Geocoding, "NaN" values return Coordinates which should not be.


tweets_df.shape #Get number of rows and columns

tweets_df.duplicated(subset='tweet_id').sum() # Check for duplicate values


tweets_df=tweets_df.drop_duplicates(subset=['tweet_id']) # drop duplicate values
#tweets_df=tweets_df.drop_duplicates() # drop duplicate values

tweets_df.shape # Check the shape after dropping duplicates

tweets_df.isna().any() # Check for "NaN" values

tweets_df['location']=tweets_df['location'].fillna('No location') # Replace "NaN" values with "No Location"

tweets_df.isna().any() # Check for "NaN" values again

#tweets_df.drop(['Location_data','location','Location'], axis=1, inplace=True)

tweets_df.columns=['Tweet_ID','Time_Created','Tweet','Location','Retweet_Count','Favorite_Count']



tweets_df.head() # Check dataframe first 5 rows

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



def getAdjectives(tweet):
    tweet = word_tokenize(tweet)  # convert string to tokens
    tweet = [word for (word, tag) in pos_tag(tweet)
             if tag == "JJ"]  # pos_tag module in NLTK library
    return " ".join(tweet)  # join words with a space in between them


# Defining my NLTK stop words and my user-defined stop words
stop_words = list(stopwords.words('english'))
user_stop_words = ['2022', 'IPL', 'ipl', 'much', 'next', 'cant', 'wont', 'hadnt',
                    'havent', 'hasnt', 'isnt', 'shouldnt', 'couldnt', 'wasnt', 'werent',
                    'mustnt', '’', '...', '..', '.', '.....', '....', 'been…', 'exciting', 'boring',
                    'interesting', 'year', 'been', 'next']
alphabets = list(string.ascii_lowercase)
stop_words = stop_words + user_stop_words + alphabets
word_list = words.words()  # all words in English language
emojis = list(UNICODE_EMOJI.keys())  # full list of emojis


# ### An example of how the above function works is shown below


print(preprocessTweets("IPL 2022 was a not very exciting watch. Hope next year the ipl will be worth watching"))


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


# Extract all tweets into one long string with each word separate with a "space"
tweets_long_string = tweets_df['Tweets_Adjectives'].tolist()
tweets_long_string = " ".join(tweets_long_string)


# Import Twitter Logo
image = np.array(Image.open('twitter1.png'))
    
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
