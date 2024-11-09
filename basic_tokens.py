# Jacqueline Silver
# this project is adapted from a class assignment in COMP 345 at McGill 
# this file does basic tokenizing of tweets
# as well as finding most frequent occuring excerpts


from google.colab import drive
drive.mount('/content/drive/')


!ls '/content/drive/My Drive/tweets'

#opens file and prints top 10 tweets

f = open("/content/drive/My Drive/tweets/20000_tweets.txt", "r")

line_count = 0
top_tweets = []
for tweet in f:
  print("Tweet ", line_count, ": ")
  print(tweet)

  top_tweets.append(tweet)
  line_count += 1
  if line_count >= 10:
    break
f.close()

"""
Basic Word Tokenizer

"""


import re

#tokenize text into words
def tokenize(text):
  words = []

  
  i = 0 #keep track of last char of token
  start = 0 #keep track of first char of token
  while i < len(text): #while the last character doesnt reach the end of the tweet

    if (text[i] == '.' or text[i] == ',' or text[i] == '!' or text[i] == '-' or text[i] == ';' or text[i] == ':'):
      words.append(text[start : i]) #add word up until punc
      words.append(text[i]) #add punc
      if (text[i] == "-"):
        start = i +1 #same as below but only 1 if it is a -
      else:
        start = i + 2 #once token is added, start becomes where this one ended
      i += 2 #skip over space following punc
      continue
    elif text[i] == "'" or text[i] == "’": #if ' for contractions
      if text[i + 1] == 'l' or text[i + 1] == 'v' or text[i + 1] == 'r' : #for 'll and 've and 're
        words.append(text[start : i]) #add word before contraction
        words.append(text[i : i + 3]) #adds contraction
        start = i + 4
        i += 1
        continue
      elif text[i - 1] == 'n' : #for n't
        words.append(text[start : i ]) #add word before
        words.append(text[i - 1 : i + 1]) #add n't
        start = i + 2
        i += 1
        continue
      elif text[i + 1] == 'm' or text[i + 1] == 's': #for 'm and 's
        words.append(text[start : i]) #add word before contraction
        words.append(text[i : i + 2]) #adds 'm
        start = i + 2
        i += 1
        continue

    elif text[i] == ' ': #if theres a space
      words.append(text[start : i]) #add word up until the space
      start = i + 1
      i += 1
      continue

    elif text[i] == 'h' and text[i + 1] == 't':
      words.append(text[start : len(text)-1])
      i += 1
      break

    elif i == (len(text) - 1): #if its the last word
      words.append(text[start : i])
      break


    i += 1 #move to next char

  for char in words:
    if char == "":
      words.remove(char)


  return words


tokenized_top_tweets = [tokenize(tweet) for tweet in top_tweets]
for tokenized_tweet in tokenized_top_tweets:
  print(tokenized_tweet)

"""
clean up the tokenized tweets

"""


#clean
def clean_a_tweet(tokenized_tweet):
  clean_tokenized_tweet = []
  for i in range (len(tokenized_tweet) - 1):

    if "https" in tokenized_tweet[i]: #change link to URL
      clean_tokenized_tweet.append('URL')
      continue

    elif tokenized_tweet[i] == '.' or tokenized_tweet[i] == '!' or tokenized_tweet[i] == '-'  or tokenized_tweet[i] == ',' or tokenized_tweet[i] == ';' or tokenized_tweet[i] == ':':
       continue #dont add puncs to new list

    elif "@" in tokenized_tweet[i]:
      clean_tokenized_tweet.append('@USER') #change @ to USER

    elif tokenized_tweet[i] == "t" or tokenized_tweet[i] == "co":
      continue


    else:
      clean_tokenized_tweet.append(tokenized_tweet[i].lower()) #do the rest of the words in lowercase

  return(clean_tokenized_tweet)

anonymized_top_tweets = [clean_a_tweet(tokenized_tweet) for tokenized_tweet in tokenized_top_tweets]
for tokenized_tweet in anonymized_top_tweets:
  print(tokenized_tweet)

"""
get the top unigrams and bigrams from the tweets

"""

import re
from collections import Counter

f = open("/content/drive/My Drive/tweets/20000_tweets.txt", "r") #open file

tweets = []

for tweet in f:
  tweets.append(tweet)

f.close

tokenized_top_tweets = [tokenize(tweet) for tweet in tweets]
anonymized_top_tweets = [clean_a_tweet(tokenized_tweet) for tokenized_tweet in tokenized_top_tweets]
all_words = []
all_2words = []

for tweet in anonymized_top_tweets:
  for word in tweet:
    all_words.append(word) #make all the words of the tweets into one long list of strings

for tweet in anonymized_top_tweets:
  for i in range (len(tweet) - 2):
    string1 = tweet[i]
    string2 = tweet[i+1]
    both = string1 + " " + string2
    all_2words.append(both) #make list of all the sets of 2 words


top_words = Counter(all_words).most_common(10) #find most common for each of the sets
top_2words = Counter(all_2words).most_common(10)

#print some results 
print("Top 10 Unigrams" + "\n") 
for tup in top_words:
  print(tup[0] + "    " + str(tup[1]))
  print("\n")

print("Top 10 Bigrams" + "\n")
for tup in top_2words:
  print(tup[0] + "    " + str(tup[1]))
  print("\n")


import re
from collections import Counter

#get the counts of specific unigrams or bigrams
allcounts = Counter(all_words)
all_2counts = Counter(all_2words)
#print results
print("covid" + " " + str(allcounts["covid"]) + "\n")
print("coronavirus" + " " + str(allcounts["coronavirus"]) + "\n")
print("republicans" + " " + str(allcounts["republicans"]) + "\n")
print("democrats" + " " + str(allcounts["democrats"]) + "\n")
print("social distancing" + " " + str(all_2counts["social distancing"]) + "\n")
print("wear mask" + " " + str(all_2counts["wear mask"]) + "\n")
print("no mask" + " " + str(all_2counts["no mask"]) + "\n")
print("donald trump" + " " + str(all_2counts["donald trump"]) + "\n")
print("joe biden" + " " + str(all_2counts["joe biden"]) + "\n")