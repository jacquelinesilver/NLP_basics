# Jacqueline Silver
# this project is adapted from a class assignment in COMP 345 at McGill 
# this project examines linguistic classifications of various languages

from google.colab import drive
drive.mount('/content/drive/')

import pandas as pd
wordforms=pd.read_csv("/content/drive/My Drive/northeuralex.csv")
display(wordforms)

#show data
languages=pd.read_csv("/content/drive/My Drive/northeuralex-languages.csv")
display(languages)

concepts=pd.read_csv("/content/drive/My Drive/northeuralex-concepts.csv")
display(concepts)

# iso code in langs = language _id
# nelex = concept id
import pandas as pd
wordforms=pd.read_csv("/content/drive/My Drive/northeuralex.csv")
languages=pd.read_csv("/content/drive/My Drive/northeuralex-languages.csv")
concepts=pd.read_csv("/content/drive/My Drive/northeuralex-concepts.csv")

languages.rename(columns={"iso_code": "Language_ID"}, inplace = True)
concepts.rename(columns={"id_nelex": "Concept_ID"}, inplace = True) #rename the columns


# merge the data frames
temp = wordforms.merge(languages, how='right')
wordforms = temp.merge(concepts, how='right') #merge dataframes
display(wordforms)

!pip install lingpy

# focus on Indo-European languages

wordforms = wordforms[(wordforms['family'] == 'Indo-European')] #only those in indoeuropean family
# only include ranks 20 and less
wordforms = wordforms[(wordforms['position_in_ranking']).isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20])] #only if rank position is = or less than 20
display(wordforms)


from re import S
import lingpy as lp
import numpy as np


#initialize confusion matrix
language_list = [] # Initialise list of languages in the current modified wordforms dataset
for lang in wordforms['name'].tolist():
  if lang not in language_list: #get only unique languages
    language_list.append(lang)
confusion = [[0 for j in range(len(language_list))] for i in range(len(language_list))]

concept_list = [] # Initialise list of languages in the current modified wordforms dataset
for con in wordforms['Concept_ID'].tolist():
  if con not in concept_list: #only unique concepts
    concept_list.append(con)

#print(len(confusion), len(confusion[0]))

for i, language1 in enumerate(language_list): #go thru each lang
  for j, language2 in enumerate(language_list):
    distances_to_avg = [] #intiialize list of distances for the words in lang1 and lang2
    for concept in concept_list: #for each concept look at the words from both langs for this concept
      conceptdistances_to_avg = [] #initialize list to avg for just this concept
      for word1 in ((wordforms[(wordforms['name'] == language1) & (wordforms['Concept_ID'] == concept)])['IPA']).tolist():
        for word2 in ((wordforms[(wordforms['name'] == language2) & (wordforms['Concept_ID'] == concept)])['IPA']).tolist():
          distance = lp.align.pairwise.edit_dist(word1, word2, normalized = True)
          conceptdistances_to_avg.append(distance) #get and add distance to list for concept
      conceptaverage = sum(conceptdistances_to_avg) / len(conceptdistances_to_avg) #get average (if theres more than one word for the concept otherwise will j be value)
      distances_to_avg.append(conceptaverage) #add this to list of distances to be averaged for the langs
    if len(distances_to_avg) != 0: #take average of the distances from the 2 langs
      average = sum(distances_to_avg) / len(distances_to_avg) #take avg of all the distances
    else:
      average = 0 #avoid divide by 0 error
    confusion[i][j] = average #set matrix
#print(confusion)


# build trees based on linguistic relationship
lp.algorithm.clustering.flat_cluster('upgma', 0.6, confusion, language_list)


#build dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

linked = linkage(confusion, 'average') #use average method on the matrix

#plot the results using dendrogram
def llf(id): return language_list[id]
plt.figure(figsize=(18, 8)) #i made it a bit wider so the labels wouldnt overlap
dendrogram(linked,
           p=100,
           truncate_mode="level",
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           leaf_label_func=llf)

plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

linked = linkage(confusion, 'single') #use average method on the matrix

#plot the results using dendrogram
def llf(id): return language_list[id]
plt.figure(figsize=(18, 8)) #i made it a bit wider so the labels wouldnt overlap
dendrogram(linked,
           p=100,
           truncate_mode="level",
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           leaf_label_func=llf)

plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt


linked = linkage(confusion, 'weighted') #use average method on the matrix

#plot the results using dendrogram
def llf(id): return language_list[id]
plt.figure(figsize=(18, 8)) #i made it a bit wider so the labels wouldnt overlap
dendrogram(linked,
           p=100,
           truncate_mode="level",
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           leaf_label_func=llf)

plt.show()


wordforms = wordforms[(wordforms['family'] == 'Indo-European')] #only those in indoeuropean family
#Problem 2b: Filter the concepts to include those less than or equal to rank 20 in the dataframe.
# your code here
wordforms = wordforms[(wordforms['position_in_ranking']) < 80] #only if rank position is = or less than 20
display(wordforms)

from re import S
import lingpy as lp
import numpy as np

#initialize confusion matrix
language_list = [] # Initialise list of languages in the current modified wordforms dataset
for lang in wordforms['name'].tolist():
  if lang not in language_list: #get only unique languages
    language_list.append(lang)
confusion = [[0 for j in range(len(language_list))] for i in range(len(language_list))]

concept_list = [] # Initialise list of languages in the current modified wordforms dataset
for con in wordforms['Concept_ID'].tolist():
  if con not in concept_list: #only unique concepts
    concept_list.append(con)
# your code here; some code to get you started:
#print(len(confusion), len(confusion[0]))

for i, language1 in enumerate(language_list): #go thru each lang
  for j, language2 in enumerate(language_list):
    distances_to_avg = [] #intiialize list of distances for the words in lang1 and lang2
    for concept in concept_list: #for each concept look at the words from both langs for this concept
      conceptdistances_to_avg = [] #initialize list to avg for just this concept
      for word1 in ((wordforms[(wordforms['name'] == language1) & (wordforms['Concept_ID'] == concept)])['IPA']).tolist():
        for word2 in ((wordforms[(wordforms['name'] == language2) & (wordforms['Concept_ID'] == concept)])['IPA']).tolist():
          distance = lp.align.pairwise.edit_dist(word1, word2, normalized = True)
          conceptdistances_to_avg.append(distance) #get and add distance to list for concept
      conceptaverage = sum(conceptdistances_to_avg) / len(conceptdistances_to_avg) #get average (if theres more than one word for the concept otherwise will j be value)
      distances_to_avg.append(conceptaverage) #add this to list of distances to be averaged for the langs
    if len(distances_to_avg) != 0: #take average of the distances from the 2 langs
      average = sum(distances_to_avg) / len(distances_to_avg) #take avg of all the distances
    else:
      average = 0 #avoid divide by 0 error
    confusion[i][j] = average #set matrix
#print(confusion)

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

linked = linkage(confusion, 'average') #use average method on the matrix

#plot the results using dendrogram
def llf(id): return language_list[id]
plt.figure(figsize=(18, 8)) #i made it a bit wider so the labels wouldnt overlap
dendrogram(linked,
           p=100,
           truncate_mode="level",
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           leaf_label_func=llf)

plt.show()



from re import S
import lingpy as lp
import numpy as np


#initialize confusion matrix
language_list = [] # Initialise list of languages in the current modified wordforms dataset
for lang in wordforms['name'].tolist():
  if lang not in language_list: #get only unique languages
    language_list.append(lang)
confusion2 = [[0 for j in range(len(language_list))] for i in range(len(language_list))]

concept_list = [] # Initialise list of languages in the current modified wordforms dataset
for con in wordforms['Concept_ID'].tolist():
  if con not in concept_list: #only unique concepts
    concept_list.append(con)

#print(len(confusion), len(confusion[0]))

for i, language1 in enumerate(language_list): #go thru each lang
  for j, language2 in enumerate(language_list):
    distances_to_avg = [] #intiialize list of distances for the words in lang1 and lang2
    for concept in concept_list: #for each concept look at the words from both langs for this concept
      conceptdistances_to_avg = [] #initialize list to avg for just this concept
      for word1 in ((wordforms[(wordforms['name'] == language1) & (wordforms['Concept_ID'] == concept)])['subfamily']).tolist():
        for word2 in ((wordforms[(wordforms['name'] == language2) & (wordforms['Concept_ID'] == concept)])['subfamily']).tolist():
          distance = lp.align.pairwise.edit_dist(word1, word2, normalized = True)
          conceptdistances_to_avg.append(distance) #get and add distance to list for concept
      conceptaverage = sum(conceptdistances_to_avg) / len(conceptdistances_to_avg) #get average (if theres more than one word for the concept otherwise will j be value)
      distances_to_avg.append(conceptaverage) #add this to list of distances to be averaged for the langs
    if len(distances_to_avg) != 0: #take average of the distances from the 2 langs
      average = sum(distances_to_avg) / len(distances_to_avg) #take avg of all the distances
    else:
      average = 0 #avoid divide by 0 error
    confusion2[i][j] = average #set matrix
#print(confusion)

from re import S
import lingpy as lp
import numpy as np

#initialize confusion matrix
language_list = [] # Initialise list of languages in the current modified wordforms dataset
for lang in wordforms['name'].tolist():
  if lang not in language_list: #get only unique languages
    language_list.append(lang)
confusion1 = [[0 for j in range(len(language_list))] for i in range(len(language_list))]

concept_list = [] # Initialise list of languages in the current modified wordforms dataset
for con in wordforms['Concept_ID'].tolist():
  if con not in concept_list: #only unique concepts
    concept_list.append(con)

for i, language1 in enumerate(language_list): #go thru each lang
  for j, language2 in enumerate(language_list):
    distances_to_avg = [] #intiialize list of distances for the words in lang1 and lang2
    for concept in concept_list: #for each concept look at the words from both langs for this concept
      conceptdistances_to_avg = [] #initialize list to avg for just this concept
      for word1 in ((wordforms[(wordforms['name'] == language1) & (wordforms['Concept_ID'] == concept)])['family']).tolist():
        for word2 in ((wordforms[(wordforms['name'] == language2) & (wordforms['Concept_ID'] == concept)])['family']).tolist():
          distance = lp.align.pairwise.edit_dist(word1, word2, normalized = True)
          conceptdistances_to_avg.append(distance) #get and add distance to list for concept
      conceptaverage = sum(conceptdistances_to_avg) / len(conceptdistances_to_avg) #get average (if theres more than one word for the concept otherwise will j be value)
      distances_to_avg.append(conceptaverage) #add this to list of distances to be averaged for the langs
    if len(distances_to_avg) != 0: #take average of the distances from the 2 langs
      average = sum(distances_to_avg) / len(distances_to_avg) #take avg of all the distances
    else:
      average = 0 #avoid divide by 0 error
    confusion1[i][j] = average #set matrix
#print(confusion)


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# create clusters based on each criteria 
q5clust = lp.algorithm.clustering.flat_cluster('upgma', 0.6, confusion, language_list)
fam = lp.algorithm.clustering.flat_cluster('upgma', 0.6, confusion1, language_list)
subfam = lp.algorithm.clustering.flat_cluster('upgma', 0.6, confusion2, language_list)

q5 = [0 for i in range(len(language_list))] #initialize lists for cluster values for each lang
family = [0 for i in range(len(language_list))]
subfamily = [0 for i in range(len(language_list))]


for i, lang in enumerate(language_list): #i terate thru langs to get the cluster key for each
  for j, clust in enumerate(q5clust.keys()):
    if lang in q5clust[clust]: # find the cluster that each lang is in the cluster
      q5[i] = j
      continue
    else:
      continue

for i, lang in enumerate(language_list): # repeat for both family and subfamily clusters
  for j, clust in enumerate(fam.keys()):
    if lang in fam[clust]:
      family[i] = j
      continue
    else:
      continue

for i, lang in enumerate(language_list):
  for j, clust in enumerate(subfam.keys()):
    if lang in subfam[clust]:
      subfamily[i] = j
      continue
    else:
      continue

v_measure_family = v_measure_score(family, q5) # then get measures from the lists that we created with the indices of the clusters for each lang
v_measure_subfamily = v_measure_score(subfamily, q5)

# save the two V measure scores as v_measure_family and v_measure_subfamily

print(v_measure_family)
print(v_measure_subfamily)