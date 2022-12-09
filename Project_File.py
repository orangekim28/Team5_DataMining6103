#%% [markdown]
#
# INTRODUCTION TO DATA MINING
# ## Credit Card Fraud Detection
#

#%%
# Importing the libraries

import numpy as np 
import pandas as pd

#%%
# Load the Credit Card Dataset into a DataFrame (CCFD_DATA)

CCFD_DATA =pd.read_csv('creditcard.csv')
CCFD_DATA.shape

#%%
# Analyzing the DataFrame : Viewing the Data

CCFD_DATA.head()

#%%
# Info about the data

CCFD_DATA.info()

#%%
# Cleaning Data : Checking for Empty Cells

CCFD_DATA.isnull().values.sum()

#%%
# Checking for Duplicate Cells

CCFD_DATA.duplicated().sum()

#%%
# Removing Duplicates

CCFD_DATA.drop_duplicates(keep=False,inplace=True)

#%%
# Re-checking for Duplicate Cells

CCFD_DATA.duplicated().sum()

#%%
# let's check the class column

CCFD_DATA['Class'].value_counts()

#%%
# Column Data Type Assessment

CCFD_DATA.dtypes.value_counts()

# %%

non_fraud = len(CCFD_DATA[CCFD_DATA.Class == 0])
fraud = len(CCFD_DATA[CCFD_DATA.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100

print("Number of Genuine transactions: ", non_fraud)
print("Number of Fraud transactions: ", fraud)
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))

# %%
# Summary Statistics

CCFD_DATA.describe()

# %%
# Exploratory Data Analysis
# Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

# %%
CCFD_DATA.hist(figsize=(20,20),color='lime')
plt.show()

# %%

CCFD_DATA['Class'].value_counts().plot(kind='bar')
plt.xticks([0,1],['Genuine', 'Fraud'])


# %%
## Since the data is highly imbalanced, we will move ahead with Undersampling/Upsampling.
### But inorder to see which method works for the data, let's create model using upsampled and undersampled data.
## We need balanced data!!
# class count
class_count_0, class_count_1 = CCFD_DATA['Class'].value_counts()

# Separate class
class_0 = CCFD_DATA[CCFD_DATA['Class'] == 0]
class_1 = CCFD_DATA[CCFD_DATA['Class'] == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)
      
      
      
      
      
      
      
# %%
# Undersampling Technique
class_0_under = class_0.sample(class_count_1)
data_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and0:",data_under['Class'].value_counts())
# plot the count after under-sampeling
data_under['Class'].value_counts().plot(kind='bar', title='count (target)')

# %%

## correlation matrix.

plt.figure(figsize=(10,8))
corr=data_under.corr()
sns.heatmap(corr,cmap='BuPu')

# %%

# Splitting the Data
from sklearn.model_selection import train_test_split

X=data_under.drop(['Class'],axis=1)
y=data_under['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)

# %%

## Model 1
## Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Fit and predict
rfc = RandomForestClassifier() 
rfc.fit(X_train, y_train) 
y_pred = rfc.predict(X_test)

# For the performance let's use some metrics from SKLEARN module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  
print("The accuracy is", accuracy_score(y_test, y_pred)) 
print("The precision is", precision_score(y_test, y_pred))
print("The recall is", recall_score(y_test, y_pred))
print("The F1 score is", f1_score(y_test, y_pred))

# %%

## Model 2
## Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred_2=lr.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_2)) 
print("The precision is", precision_score(y_test, y_pred_2))
print("The recall is", recall_score(y_test, y_pred_2))
print("The F1 score is", f1_score(y_test, y_pred_2))


# classification report
print(classification_report(y_test, y_pred_2))
# confusion matrix
fig, ax = plt. subplots ()
sns.heatmap (confusion_matrix(y_test, y_pred_2, normalize='true'), annot=True, ax=ax)
ax.set_title ("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%

## Model 3
## Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_3=dt.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_3)) 
print("The precision is", precision_score(y_test, y_pred_3))
print("The recall is", recall_score(y_test, y_pred_3))
print("The F1 score is", f1_score(y_test, y_pred_3))

# %%
