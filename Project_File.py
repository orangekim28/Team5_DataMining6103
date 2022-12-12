#%% [markdown]
#
# INTRODUCTION TO DATA MINING
# ## Credit Card Fraud Detection
#

#%%
# Importing the libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

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
## Since the data is highly imbalanced, we will move ahead with Undersampling/Upsampling.
### But inorder to see which method works for the data, let's create model using upsampled and undersampled data.
## We need balanced data!!
# class count
class_count_0, class_count_1 = CCFD_DATA['Class'].value_counts()

# Separate class
# It takes value 1 in case of fraud and 0 otherwise
class_0 = CCFD_DATA[CCFD_DATA['Class'] == 0]
class_1 = CCFD_DATA[CCFD_DATA['Class'] == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

#
#%%
# EDA for Unbalanced Data
CCFD_DATA.hist(figsize=(20,20),color='blue')
plt.show()

# %%
# Correlation matrix before Sampling
plt.figure(figsize=(10,8))
corr=CCFD_DATA.corr()
sns.heatmap(corr,cmap='BuPu')

# %%
# Splitting the Data before Sampling
from sklearn.model_selection import train_test_split

X=CCFD_DATA.drop(['Class'],axis=1)
y=CCFD_DATA['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)

# %%
# Undersampling Technique
class_0_under = class_0.sample(class_count_1)
data_under = pd.concat([class_0_under, class_1], axis=0)
print("total class of 1 and 0:",data_under['Class'].value_counts())

# plot the count after under-sampeling
data_under['Class'].value_counts().plot(kind='bar', title='count (target)')

#%%
data_under.hist(figsize=(20,20),color='violet')
plt.show()

#
#%%
# Oversampling Technique
class_1_over = class_1.sample(class_count_0, replace = True)
data_over = pd.concat([class_1_over, class_0], axis=0)
print("Total class of 1 and 0:", data_under['Class'].value_counts())

# plot the count after over-sampeling
data_over['Class'].value_counts().plot(kind='bar', title='count')

#
# %%
# Exploratory Data Analysis for Oversampling

data_over.hist(figsize=(20,20),color='green')
plt.show()

#%%
plt.hist(data_over.Time, label='time', edgecolor='black', linewidth=1)
plt.xlabel('Time (in seconds)')
plt.ylabel('Rel freq.')
plt.show()

#
#%%
plt.hist(data_over.Amount, label='time', edgecolor='black', linewidth=1)
plt.xlabel('Amount')
plt.ylabel('Rel freq.')
plt.show()

#
#%%
genuine = data_under[data_under['Class'] == 0]
fraud = data_under[data_under['Class'] == 1]

rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount vs Time of transaction for Undersampling')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in seconds)')
plt.ylabel('Amount')
plt.show()

#%%
genuine = data_over[data_over['Class'] == 0]
fraud = data_over[data_over['Class'] == 1]

rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount vs Time of transaction for Oversampling')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in seconds)')
plt.ylabel('Amount')
plt.show()

# %%
# Correlation matrix
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
# Model 1
## Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 

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

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 2
## Logistic Regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred_2=lr.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_2)) 
print("The precision is", precision_score(y_test, y_pred_2))
print("The recall is", recall_score(y_test, y_pred_2))
print("The F1 score is", f1_score(y_test, y_pred_2))

# Classification report
print(classification_report(y_test, y_pred_2))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_2, normalize='true'), annot=True, ax=ax)
ax.set_title ("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 3
## Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_3=dt.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_3)) 
print("The precision is", precision_score(y_test, y_pred_3))
print("The recall is", recall_score(y_test, y_pred_3))
print("The F1 score is", f1_score(y_test, y_pred_3))

# Classification report
print(classification_report(y_test, y_pred_3))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_3, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 4
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

neighbours = np.arange(1,25)
train_accuracy =np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i,k in enumerate(neighbours):
    #Setup a knn classifier with k neighbors
    knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)
    
    #Fit the model
    knn.fit(X_train,y_train.ravel())
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train.ravel())
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test.ravel()) 

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]

#k_nearest_neighbours_classification
knn=KNeighborsClassifier(n_neighbors=x[0],algorithm="kd_tree",n_jobs=-1)
knn.fit(X_train,y_train.ravel())

# %%

knn = KNeighborsClassifier(n_neighbors = 4) 
knn.fit(X_train,y_train)
y_pred_4 = knn.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_4)) 
print("The precision is", precision_score(y_test, y_pred_4))
print("The recall is", recall_score(y_test, y_pred_4))
print("The F1 score is", f1_score(y_test, y_pred_4))

# Classification report
print(classification_report(y_test, y_pred_4))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_4, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")


# %%
# Splitting the Data : Over Sampling
from sklearn.model_selection import train_test_split

X=data_over.drop(['Class'],axis=1)
y=data_over['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)

# %%
# Model 1
## Random Forest Classifier : Over Sampling

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

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 2
## Logistic Regression : Over Sampling

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred_2=lr.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_2)) 
print("The precision is", precision_score(y_test, y_pred_2))
print("The recall is", recall_score(y_test, y_pred_2))
print("The F1 score is", f1_score(y_test, y_pred_2))

# Classification report
print(classification_report(y_test, y_pred_2))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_2, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 3
## Decision Tree : Over Sampling
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_3=dt.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_3)) 
print("The precision is", precision_score(y_test, y_pred_3))
print("The recall is", recall_score(y_test, y_pred_3))
print("The F1 score is", f1_score(y_test, y_pred_3))

# Classification report
print(classification_report(y_test, y_pred_3))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_3, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
# Model 4
# KNN Model : Over Sampling

neighbours = np.arange(1,25)
train_accuracy =np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i,k in enumerate(neighbours):
    #Setup a knn classifier with k neighbors
    knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)
    
    #Fit the model
    knn.fit(X_train,y_train.ravel())
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train.ravel())
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test.ravel()) 

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]

#k_nearest_neighbours_classification
knn=KNeighborsClassifier(n_neighbors=x[0],algorithm="kd_tree",n_jobs=-1)
knn.fit(X_train,y_train.ravel())

# %%

knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(X_train,y_train)
y_pred_4 = knn.predict(X_test)

print("The accuracy is", accuracy_score(y_test, y_pred_4)) 
print("The precision is", precision_score(y_test, y_pred_4))
print("The recall is", recall_score(y_test, y_pred_4))
print("The F1 score is", f1_score(y_test, y_pred_4))

# Classification report
print(classification_report(y_test, y_pred_4))

# Confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_4, normalize='true'), annot=True, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_ylabel("Real Value")
ax.set_xlabel("Predicted")

# %%
