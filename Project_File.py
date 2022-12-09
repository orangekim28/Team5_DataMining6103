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

test_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and0:",test_under['Class'].value_counts())
# plot the count after under-sampeling
test_under['Class'].value_counts().plot(kind='bar', title='count (target)')

# %%
