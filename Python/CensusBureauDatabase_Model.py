# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:05:11 2018

@author: Nevena Mitic
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib
########################## Preparation of Data Set ####################################
"""
1. Source: 
   This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html
2. Number of Instances: 32561
3. Number of Attributes: 14
4. Attribute Information:
   - age: continuous.
   - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
   - fnlwgt: continuous.
   - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
   - education-num: continuous.
   - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
   - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
   - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
   - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
   - sex: Female, Male.
   - capital-gain: continuous.
   - capital-loss: continuous.
   - hours-per-week: continuous.
   - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

"""

print("This data was extracted from the census bureau database found at http://www.census.gov/ftp/pub/DES/www/welcome.html")
# Read in data as a pandas data frame
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Adults = pd.read_csv(url,sep=',', header=None)

Adults.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
               'capital-loss', 'hours-per-week', 'native-country', 'target']

# Check number of rows and columns
print("\nNumber of rows and columns:")
print(Adults.shape)

# Check the data types
print("\nData types:")
print(Adults.dtypes)

# Function for outlier replacement
def account_outliers(df, columnName):
    LimitHi = np.mean(df.loc[:, columnName]) + 2*np.std(df.loc[:, columnName])
    LimitLo = np.mean(df.loc[:, columnName]) - 2*np.std(df.loc[:, columnName])
    FlagBad = (df.loc[:, columnName] < LimitLo) | (df.loc[:, columnName] > LimitHi)
    if(any(FlagBad)):
      df.loc[FlagBad, columnName]= np.median(df.loc[:, columnName])

# Function for median imputation of the missing numeric values
def replace_missing(df, columnName):
    df.loc[:, columnName] = pd.to_numeric(df.loc[:, columnName], errors='coerce')
    HasNan = np.isnan(df.loc[:,columnName])
    df.loc[HasNan, columnName] = np.nanmedian(df.loc[:,columnName]) 
    
# Function for distributin of numerical variable
def distribution_numerical_variable(df, columnName):
    plt.title("Distribution of " + columnName + " column")
    plt.hist(df.loc[:, columnName])
    plt.show()
    
# Function for distributin of categorical variable
def distribution_categorical_variable(df, columnName):
    plt.title("Distribution of " + columnName + " column")
    df.loc[:,columnName].value_counts().plot(kind='bar')   
    plt.show()
    
   
# Removing outliers and Imputing missing values for numerical variables "age", "fnlwgt", "education-num",
    # "capital-gain", "capital-loss", "hours-per-week"    
replace_missing(Adults, "age")
replace_missing(Adults, "fnlwgt")
replace_missing(Adults, "education-num")
replace_missing(Adults, "capital-gain")
replace_missing(Adults, "capital-loss")
replace_missing(Adults, "hours-per-week")
account_outliers(Adults, "age")
account_outliers(Adults, "fnlwgt")
account_outliers(Adults, "education-num")
account_outliers(Adults, "capital-gain")
account_outliers(Adults, "capital-loss")
account_outliers(Adults, "hours-per-week")


# Distribution of numerical variables
distribution_numerical_variable(Adults, "age")
distribution_numerical_variable(Adults, "fnlwgt")
distribution_numerical_variable(Adults, "education-num")
distribution_numerical_variable(Adults, "capital-gain")
distribution_numerical_variable(Adults, "capital-loss")
distribution_numerical_variable(Adults, "hours-per-week")

# Strip whitespace
Adults['education'] = Adults["education"].map(str.strip)
Adults['target'] = Adults["target"].map(str.strip)
Adults['workclass'] = Adults["workclass"].map(str.strip)
Adults['occupation'] = Adults["occupation"].map(str.strip)

# Simplify workclass by consolidate data. 
# Missing values are imputed
Adults.loc[Adults.loc[:, "workclass"] == "?", "workclass"] = "Without-pay"
Adults.loc[Adults.loc[:, "workclass"] == "Self-emp-not-inc", "workclass"] = "Private"
Adults.loc[Adults.loc[:, "workclass"] == "Self-emp-inc", "workclass"] = "Private"
Adults.loc[Adults.loc[:, "workclass"] == "Local-gov", "workclass"] = "Federal-gov"
Adults.loc[Adults.loc[:, "workclass"] == "State-gov", "workclass"] = "Federal-gov"
Adults.loc[Adults.loc[:, "workclass"] == "Never-worked", "workclass"] = "Without-pay"
Adults.loc[Adults.loc[:, "occupation"] == "?", "occupation"] = "Other-service"

# Consolidate education levels into 3 categories, low, mid, high
Adults.loc[Adults.loc[:, "education"] == "Bachelors", "education"] = "HIGH"
Adults.loc[Adults.loc[:, "education"] == "Some-college", "education"] = "HIGH"
Adults.loc[Adults.loc[:, "education"] == "11th", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "HS-grad", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "Prof-school", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "Assoc-acdm", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "Assoc-voc", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "9th", "education"] = "LOW"
Adults.loc[Adults.loc[:, "education"] == "7th-8th", "education"] = "LOW"
Adults.loc[Adults.loc[:, "education"] == "12th", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "Masters", "education"] = "HIGH"
Adults.loc[Adults.loc[:, "education"] == "1st-4th", "education"] = "LOW"
Adults.loc[Adults.loc[:, "education"] == "Doctorate", "education"] = "HIGH"
Adults.loc[Adults.loc[:, "education"] == "10th", "education"] = "MID"
Adults.loc[Adults.loc[:, "education"] == "5th-6th", "education"] = "LOW"
Adults.loc[Adults.loc[:, "education"] == "Preschool", "education"] = "LOW"

# Consolidate occupation data
Adults.loc[Adults.loc[:, "occupation"] == "Tech-support", "occupation"] = "OfficeWork"
Adults.loc[Adults.loc[:, "occupation"] == "Craft-repair", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Other-service", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Exec-managerial", "occupation"] = "OfficeWork"
Adults.loc[Adults.loc[:, "occupation"] == "Prof-specialty", "occupation"] = "OfficeWork"
Adults.loc[Adults.loc[:, "occupation"] == "Handlers-cleaners", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Machine-op-inspct", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Adm-clerical", "occupation"] = "OfficeWork"
Adults.loc[Adults.loc[:, "occupation"] == "Farming-fishing", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Transport-moving", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Priv-house-serv", "occupation"] = "Service"
Adults.loc[Adults.loc[:, "occupation"] == "Protective-serv", "occupation"] = "Service"

# Distribution of categorical variables
distribution_categorical_variable(Adults, "workclass")
distribution_categorical_variable(Adults, "education")
distribution_categorical_variable(Adults, "marital-status")
distribution_categorical_variable(Adults, "occupation")
distribution_categorical_variable(Adults, "relationship")
distribution_categorical_variable(Adults, "race")
distribution_categorical_variable(Adults, "sex")

# One-hot encoding educational column
# Create 3 new columns, one for each category
Adults.loc[:, "LOW"] = (Adults.loc[:, "education"] == "LOW").astype(int)
Adults.loc[:, "MID"] = (Adults.loc[:, "education"] == "MID").astype(int)
Adults.loc[:, "HIGH"] = (Adults.loc[:, "education"] == "HIGH").astype(int)

# One-hot encoding work-class column
# Create 3 new columns, one for each category
Adults.loc[:, "Private"] = (Adults.loc[:, "workclass"] == "Private").astype(int)
Adults.loc[:, "Federal-gov"] = (Adults.loc[:, "workclass"] == "Federal-gov").astype(int)
Adults.loc[:, "Without-pay"] = (Adults.loc[:, "workclass"] == "Without-pay").astype(int)

# One-hot encoding occupation column
# Create 3 new columns, one for each category
Adults.loc[:, "OfficeWork"] = (Adults.loc[:, "occupation"] == "OfficeWork").astype(int)
Adults.loc[:, "Service"] = (Adults.loc[:, "occupation"] == "Service").astype(int)
Adults.loc[:, "Sales"] = (Adults.loc[:, "occupation"] == "Sales").astype(int)
Adults.loc[:, "Armed-Forces"] = (Adults.loc[:, "occupation"] == "Armed-Forces").astype(int)
##############

# Remove obsolete columns
Adults = Adults.drop("education", axis=1)
Adults = Adults.drop("workclass", axis=1)
Adults = Adults.drop("occupation", axis=1)
#print (Adults.head())

#########################################################################################

###########################  Unsupervised Learning ######################################
# Run Kmeans on the age column, and split into 4 buckets by age
columnAge = Adults.loc[:, "age"]
# Normalize attributes prior to K-Means
min = columnAge.min()
max = columnAge.max()
if max == 0:
    raise Exception("no max")
mid = (min + max) / 2

columnAge = columnAge.subtract(min).divide(max-min)
reshapedAge = np.reshape(columnAge.values, (-1, 1))
kmeans = KMeans(n_clusters=4).fit(reshapedAge)
labels_df = pd.DataFrame(kmeans.labels_)
# Adding cluster label to the data set to be used in supervised learning
Adults['age-val'] = labels_df.values

###########################  Supervised Learning #########################################

# Prediction task is to determine whether a person makes over 50K a year?

Adults.loc[Adults.loc[:, "target"] == '<=50K', "target"] = 0
Adults.loc[Adults.loc[:, "target"] == '>50K', "target"] = 1
#Adults.loc[:, "target"] = Adults.loc[:, "target"].astype('int')

# Select only feature columns and target column for classification
Adults = Adults[['age-val', 'LOW', 'MID', 'HIGH','Private', 'Federal-gov', 'Without-pay', 
                 'OfficeWork', 'Service','Sales','Armed-Forces','hours-per-week',
                  'capital-gain', 'capital-loss', 'target']]

# Split data into training and testing data set
TestFraction = 0.3
print ("Test fraction is chosen to be:", TestFraction)

print ('\nSimple approximate split:')
isTest = np.random.rand(len(Adults)) < TestFraction
TrainSet = Adults[~isTest]
TestSet = Adults[isTest] # should be 249 but usually is not
print ('Test size should have been ', 
       TestFraction*len(Adults), "; and is: ", len(TestSet))

print ('\nsklearn accurate split:')
TrainSet, TestSet = train_test_split(Adults, test_size=TestFraction)

Target = "target"
Inputs = list(Adults.columns)
Inputs.remove(Target)


Threshold = 0.5 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
# LogisticRegression Classifier #############################################################

clf = LogisticRegression() # default parameters are fine
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs]) 
# // we have probabilities of been 0 and been 1, and we are only interested in been 1
probabilities = BothProbabilities[:,1]

print ('\nConfusion Matrix and Metrics for LogisticRegression:')

predictions = (probabilities > Threshold).astype(int)
CM_LogisticRegression = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM_LogisticRegression.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR_LogisticRegression = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate for LogisticRegression:", np.round(AR_LogisticRegression, 2))
P_LogisticRegression = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision for LogisticRegression:", np.round(P_LogisticRegression, 2))
R_LogisticRegression = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall for LogisticRegression:", np.round(R_LogisticRegression, 2))

fpr, tpr, th = roc_curve(TestSet.loc[:,Target], probabilities)
AUC = auc(fpr, tpr)

# RandomForest Classifier ####################################################################
clf1 = RandomForestClassifier() # default parameters are fine
clf1.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities1 = clf1.predict_proba(TestSet.loc[:,Inputs]) 
# // we have probabilities of been 0 and been 1, and we are only interested in been 1
probabilities1 = BothProbabilities1[:,1]

print ('\nConfusion Matrix and Metrics for RandomForestClassifier:')

predictions1 = (probabilities1 > Threshold).astype(int)
CM_RandomForest = confusion_matrix(TestSet.loc[:,Target], predictions1)
tn, fp, fn, tp = CM_RandomForest.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR_RandomForest = accuracy_score(TestSet.loc[:,Target], predictions1)
print ("Accuracy rate for RandomForest:", np.round(AR_RandomForest, 2))
P_RandomForest = precision_score(TestSet.loc[:,Target], predictions1)
print ("Precision for RandomForest:", np.round(P_RandomForest, 2))
R_RandomForest = recall_score(TestSet.loc[:,Target], predictions1)
print ("Recall for RandomForest:", np.round(R_RandomForest, 2))

 # False Positive Rate, True Posisive Rate, probability thresholds
fpr1, tpr1, th1 = roc_curve(TestSet.loc[:,Target], probabilities1)
AUC1 = auc(fpr1, tpr1)

# ROC Curve LogisticRegression ################################################################################
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve (LogisticRegressionClassifier)')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

# ROC Curve RandomForest#######################################################################################
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve (RandomForestClassifier)')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr1, tpr1, LW=3, label='ROC curve (AUC = %0.2f)' % AUC1)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()
# print(Adults.head())