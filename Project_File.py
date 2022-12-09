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

CCFD_DATA=pd.read_csv('creditcard.csv')
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
