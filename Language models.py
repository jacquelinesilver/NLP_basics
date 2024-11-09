# Jacqueline Silver
# this project is adapted from a class assignment in COMP 345 at McGill
# This project explores language models and generating language

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)


!ls "/content/drive/MyDrive/data"

import json
from collections import Counter
import numpy as np
import nltk
from nltk.data import find
import gensim
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
nltk.download('word2vec_sample')

class NgramLM:
	def __init__(self):
		"""
		N-gram Language Model
		"""
		# Dictionary to store next-word possibilities for bigrams
		#Maintains a list for each bigram.
		self.bigram_prefix_to_trigram = {}

		# Dictionary to store counts of corresponding next-word possibilities for bigrams
		# Maintains a list for each bigram.
		self.bigram_prefix_to_trigram_weights = {}

	def load_trigrams(self):
		with open("/content/drive/My Drive/data/tweets/covid-tweets-2020-08-10-2020-08-21.trigrams.txt") as f:
			lines = f.readlines()
			for line in lines:
				word1, word2, word3, count = line.strip().split()
				if (word1, word2) not in self.bigram_prefix_to_trigram:
					self.bigram_prefix_to_trigram[(word1, word2)] = []
					self.bigram_prefix_to_trigram_weights[(word1, word2)] = []
				self.bigram_prefix_to_trigram[(word1, word2)].append(word3)
				self.bigram_prefix_to_trigram_weights[(word1, word2)].append(int(count))


	def top_next_word(self, word1, word2, n=10):

		next_words = []
		probs = []

		
		totalfreqbigram = 0 #initialize sum of counts
		probdic = {} #dictionary for probabilities for third words

		if (word1, word2) not in self.bigram_prefix_to_trigram.keys():
			return next_words, probs #make sure the bigram exists if not return empty lists

		else:

			for count in self.bigram_prefix_to_trigram_weights[(word1,word2)]: #add all the counts together to get a sum for probs
				totalfreqbigram += count

			for i, nextword in enumerate(self.bigram_prefix_to_trigram[(word1, word2)]):
				#iterate thru all the possible third words and find the corresponding probabilities
				probability = float((self.bigram_prefix_to_trigram_weights[(word1, word2)][i]) / totalfreqbigram)
				probdic[nextword] = probability #add the probability to the key of the 3rd word

			top_n = dict(sorted(probdic.items(), key=lambda x : x[1], reverse=True)[:n]) #sort and get the top 10 in the dictionary

			for word in top_n.keys():
				next_words.append(word) #add the top words to return word list
			for prob in top_n.values():
				probs.append(prob)

		return next_words, probs

	def sample_next_word(self, word1, word2, n=10):
		

		next_words = []
		probs = []
		
		#do this again to get probabilities
		totalfreqbigram = 0 #initialize sum of counts
		probdic = {} #dictionary for probabilities for third words

		if (word1, word2) not in self.bigram_prefix_to_trigram.keys():
			return next_words, probs #make sure the bigram exists if not return empty lists

		else:
			probabilities = []
			for count in self.bigram_prefix_to_trigram_weights[(word1,word2)]: #add all the counts together to get a sum for probs
				totalfreqbigram += count
			for weight in (self.bigram_prefix_to_trigram_weights[(word1, word2)]):
				probabilities.append(float(weight / totalfreqbigram)) #get list of probabilities
			#get n third words from trigram list w the probs w no repetition
			randowords = np.random.choice((self.bigram_prefix_to_trigram[(word1, word2)]), size=n, replace=False, p=(probabilities))
			for word in randowords: #add the words to the word return list
				next_words.append(word)

			for i, nextword in enumerate(self.bigram_prefix_to_trigram[(word1, word2)]):
				#iterate thru all the possible third words and find the corresponding probabilities
				probability = float((self.bigram_prefix_to_trigram_weights[(word1, word2)][i]) / totalfreqbigram)
				probdic[nextword] = probability #add the probability to the key of the 3rd word

			for word1 in enumerate(randowords):
				for word2 in probdic.keys(): #iterate thru random words and find in probability dic
					if (word1 == word2):
						prob = probdic[word2]
						probs.append(prob) #add probabilities to the prob return list

		return next_words, probs

	def generate_sentences(self, prefix, beam=10, sampler=top_next_word, max_len=20):
		sentences = []
		probs = []

		words_in_pre = prefix.split(" ") #get words individually from prefix

		sentencedic = {}
		probdic = {}

		for k in range(beam):
			sentencedic[k] = words_in_pre #initialize dictionary of beam # of sentences with prefix words
			probdic[k] = 1 #intialize dictionary to keep track of probabilities for sentences

		wordsleft = (max_len - (words_in_pre.length)) #get num of words before max length is hit

		#keep calling the function and each time determine the top beam# of probabilities then continue the process on those
		while wordsleft > 0:
			temp = {} #make a holding dictionary for all the possible sentences w their probs
			for i, sentence in enumerate(sentencedic.values()):
				word1 = sentence[-2] #get last bigram
				word2 = sentence[-1]
				if (word2 == 'EOS'):
					continue
				else:
					words, probs = sampler(self, word1, word2, beam) #call sample func with beam to get that num of results
					for j, word in enumerate(words):
						new_sentence = sentence.append(word) #add new 3rd word to end of sentence
						new_prob = (probs[j] * probdic[i]) #get new sentence probability w this word
						temp[new_sentence] = new_prob #add the new sentence and prob to temp


			topbeam = dict(sorted(temp.items(), key=lambda x : x[1], reverse=True)[:beam]) #sort and get the top beam#

			for l, sent in enumerate(topbeam.keys()):
				sentencedic[l] = sent #set the dictionaries to the new top beam# sentences and probabilities
				probdic[l] = topbeam[sent]

			wordsleft -= 1 #increment down until no words left

		for sentence1 in sentencedic:
			sentences.append(sentence1)
#add the final top beam# of sentences to the return lists
		for probability1 in probdic:
			probs.append(probability1)
		return sentences, probs


# Define your language model object
language_model = NgramLM()
# Load trigram data
language_model.load_trigrams()


next_words, probs = language_model.top_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)


next_words, probs = language_model.sample_next_word("middle", "of", 10)
for word, prob in zip(next_words, probs):
	print(word, prob)

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.top_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> trump", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)
print("#########################\n")

sentences, probs = language_model.generate_sentences(prefix="<BOS1> <BOS2> biden", beam=10, sampler=language_model.sample_next_word)
for sent, prob in zip(sentences, probs):
	print(sent, prob)

!ls "/content/drive/My Drive/data/semantic-parser"
parser_files = "/content/drive/My Drive/data/semantic-parser"

