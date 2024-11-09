# Jacqueline Silver
# this project is adapted from a class assignment in COMP 345 at McGill
# examines linguistic data regarding word frequency data and reaction times
# using plots and regression analytics


#open necessary data 
# throws an error if your Drive folder doesn't contain english_a4.csv
from google.colab import drive
drive.mount('/content/drive/')
!ls "/content/drive/My Drive/english_a4.csv"


import pandas as pd

#show data in csv 
english = pd.read_csv("/content/drive/My Drive/english_a4.csv")
display(english)

# manipulate data to only keep certain aspects
# only keep young speaker data for lexical decision reaction time,
# word, then word category and voice (made numerical) and other predictive categories, the rest dropped

english_young = pd.read_csv("/content/drive/My Drive/english_a4.csv")

# simplify data
# subset to young speakers

english_young = english_young[english['AgeSubject'] == 'young'] #only keep young rows

# restrict to certain columns

english_young = english_young.drop(['AgeSubject', 'RTnaming', 'NounFrequency', 'VerbFrequency', 'FrequencyInitialDiphoneWord', 'FrequencyInitialDiphoneSyllable', 'DerivationalEntropy', 'NumberSimplexSynsets', 'NumberComplexSynsets', 'ConfriendsN', 'ConffV', 'ConffN', 'ConfbV', 'ConfbN', 'CV', 'Obstruent', 'Frication', 'CorrectLexdec', 'Ncount', 'MeanBigramFrequency', 'FrequencyInitialDiphone', 'ConspelV', 'ConspelN', 'ConphonV', 'ConphonN', 'ConfriendsV'], axis = 1)
#drop all the various columns that arent word or lexdecRT or predictive

# map categorical predictors to numeric
category_to_number_dict = {'N': 0,'V': 1, 'voiced': 0,'voiceless': 1}

english_young['WordCategory'] = english_young['WordCategory'].map(category_to_number_dict)
english_young['Voice'] = english_young['Voice'].map(category_to_number_dict)

# print message
print('The numbers of rows and columns respectively are: ', english_young.shape[0], ' and ', english_young.shape[1])


display(english_young)


import seaborn as sns; sns.set()
# create plot 
sns.pairplot(english_young, kind = 'reg', diag_kind='kde')

#examine written frequency vs reaction time
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

englishdata = english_young[['WrittenFrequency', 'RTlexdec']] #get only the necessary data

plots, axes = plt.subplots(1, 2, figsize=(50,25)); #plots 1 x 2 and make bigger (more spread out)

a = sns.regplot(x="WrittenFrequency", y="RTlexdec", data=englishdata, line_kws={"color": "C1"}, ax=axes[0]);
b = sns.regplot(x="WrittenFrequency", y="RTlexdec", data=englishdata, lowess=True, line_kws={"color": "C1"}, ax=axes[1]);

a.set(xlabel="Written Frequency", ylabel="Lexical Decision Time") #clear labels
b.set(xlabel="Written Frequency", ylabel="Lexical Decision Time")


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import mean_squared_error

def bic(X, y, degree, model):
  # number of observations
  n = X.shape[0]

  # number of parameters
  k = degree + 1

  # calculate Residual Sum of Squares)
  RSS = mean_squared_error(y, model.predict(X)) * n

  BIC = n * np.log(RSS / n) + k * np.log(n)

  return(BIC)

# set up a predictor matrix X for features -- considering just the written frequency feature
# set up the outcome vector, y.

X = english_young[['WrittenFrequency']] #get matrix with only written freq column
y = english_young['RTlexdec'] #get reaction time values

# split the data into train and test subsets, with 20% of the data in test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# set up a plot of training data
X_plot = np.linspace(0, 10,5000).reshape(-1, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.1)


print("Model class: " + "Linear Regression")
for degree in [1,2,3,4,5,6,7,10,25]:
  # fit a polynomial regression model with this degree, on the training data

  r = LinearRegression()
  model = make_pipeline(PolynomialFeatures(degree),r)
  model.fit(X_train, y_train)

  print("\tDegree " + str(degree) +"\n\t\tTrain R^2: "+ str(model.score(X_train,y_train)))
  print("\t\tTest R^2: "+ str(model.score(X_test,y_test)))
  print("\t\tBIC: "+ str(bic(X_test, y_test, degree, model)))


  plt.plot(X_train, y_train, color='pink', alpha=0.1)
  #add dif color line to the plot

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#k = 4
english_young['WrittenFrequency2'] = english_young['WrittenFrequency'] ** 2
english_young['WrittenFrequency3'] = english_young['WrittenFrequency'] ** 3
english_young['WrittenFrequency4'] = english_young['WrittenFrequency'] ** 4

X = english_young[['Familiarity', 'WordCategory', 'WrittenFrequency', 'WrittenFrequency2', 'WrittenFrequency3', 'WrittenFrequency4', 'WrittenSpokenFrequencyRatio', 'FamilySize', 'InflectionalEntropy', 'Voice', 'LengthInLetters']]
y = english_young['RTlexdec']

# define X_std and Y_std:
# X_std is the X matrix above, but with each column z-scored
#y_std is the same as y above, but z-scored

X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y_2d = y.values.reshape(-1, 1)
y_std = scaler.fit_transform(y_2d)

# split the data into train and test subsets, with 20% of the data in test.
# this should define objects called X_train, X_test, y_train, and y_test.


X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=42)

from sklearn.linear_model import Lasso
# fit a Lasso linear regression to X_train and y_train (which correspond to the train split of the X_std and y_std data), with alpha parameter of 0.02.

mod_lasso=Lasso(alpha = 0.02)
mod_lasso.fit(X_std, y_std)

# display the R^2 of this model on the train and test set


print("\t\tTrain R^2: "+ str(mod_lasso.score(X_train,y_train)))
print("\t\tTest R^2: "+ str(mod_lasso.score(X_test,y_test)))

# extract model coeffs
coefficients = mod_lasso.coef_

# make the dataframe
# Intercept of the model
intercept = mod_lasso.intercept_
feature_names = X_train.columns

# sort the DataFrame by coefficient magnitude
# Your code here
coefficients_with_features = pd.DataFrame(zip(feature_names, coefficients), columns=['Feature', 'Coefficient']).sort_values("Coefficient", ascending=True)

print(coefficients_with_features)
