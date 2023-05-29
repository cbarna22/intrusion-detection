#Network Intrusion Classification Script
#5/28/23
#This script uses a Kaggle dataset of network port statistics
#to classify different types of cyber attacks: Blackhole, PortScan,
#Overflow, TCP-SYN, Diversion, and Normal. It uses the following
#data mining algorithms: Gaussian Naive Bayes, K-Nearest Neighbors,
#Decision Tree, and Random Forest.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.utils import resample #balancing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


# read the csv file
df = pd.read_csv('UNR-IDD.csv')
df = df.dropna()

#plotting classes
sns.countplot(x='Label', data = df)
plt.figure(figsize=(10,6))
plt.show()

#plotting attack vs no attack
sns.countplot(x='Binary Label', data = df)
plt.figure(figsize=(10,6))
plt.show()

#drop constant features
df = df.drop('Packets Rx Dropped', axis='columns')
df = df.drop('Packets Tx Dropped', axis='columns')
df = df.drop('Packets Rx Errors', axis='columns')
df = df.drop('Packets Tx Errors', axis='columns')
df = df.drop('Delta Packets Rx Dropped', axis='columns')
df = df.drop(' Delta Packets Tx Dropped', axis='columns')
df = df.drop('Delta Packets Rx Errors', axis='columns')
df = df.drop('Delta Packets Tx Errors', axis='columns')
df = df.drop('is_valid', axis='columns')
df = df.drop('Table ID', axis='columns')
df = df.drop('Max Size', axis='columns')
df = df.drop('Binary Label', axis='columns')

#missing values
print('Number of missing values: ')
print(df.isna().sum())


#convert string columns to floats
df['Switch ID'] = df['Switch ID'].str.extract('(\d+(?:\.\d+)?)').astype(float)
df['Port Number'] = df['Port Number'].str.extract('(\d+(?:\.\d+)?)').astype(float)



#outlier removal using quantile-based flooring and capping

def quantileFloor(str):
    df[str] = np.where(df[str] < df[str].quantile(0.001),df[str].quantile(0.001),df[str])

def quantileCap(str):
    df[str] = np.where(df[str] > df[str].quantile(0.999),df[str].quantile(0.999),df[str])

quantileFloor('Received Packets')
quantileCap('Received Packets')
quantileFloor('Received Bytes')
quantileCap('Received Bytes')
quantileFloor('Sent Bytes')
quantileCap('Sent Bytes')
quantileFloor('Sent Packets')
quantileCap('Sent Packets')
quantileFloor('Port alive Duration (S)')
quantileCap('Port alive Duration (S)')
quantileFloor('Delta Received Packets')
quantileCap('Delta Received Packets')
quantileFloor('Delta Received Bytes')
quantileCap('Delta Received Bytes')
quantileFloor('Delta Sent Bytes')
quantileCap('Delta Sent Bytes')
quantileFloor('Delta Sent Packets')
quantileCap('Delta Sent Packets')
quantileFloor('Delta Port alive Duration (S)')
quantileCap('Delta Port alive Duration (S)')
quantileFloor('Connection Point')
quantileCap('Connection Point')
quantileFloor('Total Load/Rate')
quantileCap('Total Load/Rate')
quantileFloor('Total Load/Latest')
quantileCap('Total Load/Latest')
quantileFloor('Unknown Load/Rate')
quantileCap('Unknown Load/Rate')
quantileFloor('Unknown Load/Latest')
quantileCap('Unknown Load/Latest')
quantileFloor('Latest bytes counter')
quantileCap('Latest bytes counter')
quantileFloor('Active Flow Entries')
quantileCap('Active Flow Entries')
quantileFloor('Packets Looked Up')
quantileCap('Packets Looked Up')
quantileFloor('Packets Matched')
quantileCap('Packets Matched')


#balancing data

# Separate classes
label_portscan = df[df.Label == 'PortScan']
label_tcpsyn = df[df.Label == 'TCP-SYN']
label_overflow = df[df.Label == 'Overflow']
label_blackhole = df[df.Label == 'Blackhole']
label_diversion = df[df.Label == 'Diversion']
label_normal = df[df.Label == 'Normal']



#upsampling

label_overflow_upsampled = resample(label_overflow, 
                                 replace=True,    # sample with replacement
                                 n_samples=4000,     
                                 random_state=123) # reproducible results

label_normal_upsampled = resample(label_normal, 
                                 replace=True, 
                                 n_samples=4000,
                                 random_state=123)

#downsampling

label_portscan_downsampled = resample(label_portscan, 
                                 replace=False,    # sample without replacement
                                 n_samples=4000,     
                                 random_state=123) 

label_tcpsyn_downsampled = resample(label_tcpsyn, 
                                 replace=False, 
                                 n_samples=4000,
                                 random_state=123)

label_blackhole_downsampled = resample(label_blackhole, 
                                 replace=False, 
                                 n_samples=4000,
                                 random_state=123)

label_diversion_downsampled = resample(label_diversion, 
                                 replace=False, 
                                 n_samples=4000,
                                 random_state=123)

df = pd.concat([label_overflow_upsampled, label_normal_upsampled])
df = pd.concat([df, label_portscan_downsampled, label_tcpsyn_downsampled])
df = pd.concat([df, label_blackhole_downsampled, label_diversion_downsampled])

#plotting classes
sns.countplot(x='Label', data = df)
plt.figure(figsize=(10,6))
plt.show()



y = df['Label'].values
X = df[df.columns.drop('Label')].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)


#normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# feature selection
def select_features(X_train, y_train, X_test):
 # configure to select all features
 selector = SelectKBest(score_func=mutual_info_classif, k='all')
 # learn relationship from training data
 selector.fit(X_train, y_train)
 # transform train input data
 X_train_select = selector.transform(X_train)
 # transform test input data
 X_test_select = selector.transform(X_test)
 return X_train_select, X_test_select, selector

# feature selection
X_train_select, X_test_select, selector = select_features(X_train, y_train, X_test)
# what are scores for the features
print('Feature Importance:')
for pos in range(len(selector.scores_)):
 print('Feature %d: %f' % (pos, selector.scores_[pos]))
 print(df.columns[pos])
 print()
# plot the scores
#plt.bar([pos for pos in range(len(selector.scores_))], selector.scores_)
feat_importances = pd.Series(selector.scores_)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#dropping unimportant features so we only keep 12 most important features
df = df.drop('Delta Port alive Duration (S)', axis='columns')
df = df.drop('Connection Point', axis='columns')
df = df.drop('Unknown Load/Rate', axis='columns')
df = df.drop('Latest bytes counter', axis='columns')
df = df.drop('Total Load/Rate', axis='columns')
df = df.drop('Delta Received Packets', axis='columns')
df = df.drop('Delta Sent Packets', axis='columns')
df = df.drop('Total Load/Latest', axis='columns')



#nb model
print('NB Classification Report:')
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))


#nb confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('NB Confusion Matrix', fontsize=18)


#knn model

#hyperparameter tuning
leaf_size = list(range(1,5))
n_neighbors = list(range(1,5))
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors)

hp_knn = KNeighborsClassifier()
clf = GridSearchCV(hp_knn, hyperparameters, cv=10)
best_knn = clf.fit(X_train, y_train)

#creating the knn model and displaying accuracy
print('KNN Classification Report:')
knn = KNeighborsClassifier(leaf_size = best_knn.best_estimator_.get_params()['leaf_size'],
                           n_neighbors = best_knn.best_estimator_.get_params()['leaf_size'])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))


#knn confusion matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('KNN Confusion Matrix', fontsize=18)
plt.show()

#decision tree classifier

#hyperparameter tuning
max_depth = [2,4,6,8,10,12,14,16]

hyperparameters = dict(max_depth=max_depth)

hp_dt = DecisionTreeClassifier()
clf = GridSearchCV(hp_dt, hyperparameters, cv=10)
best_dt = clf.fit(X_train, y_train)


#creating dt model and displaying accuracy
print('Decision Tree Classification Report:')
dt = DecisionTreeClassifier(max_depth = best_dt.best_estimator_.get_params()['max_depth'])
dt = dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

#decision tree confusion matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Decision Tree Confusion Matrix', fontsize=18)
plt.show()

#random forest model

#hyperparameter tuning
n_estimators = [8,10,12,14]

hyperparameters = dict(n_estimators=n_estimators)

hp_rf = RandomForestClassifier()
clf = GridSearchCV(hp_rf, hyperparameters, cv=10)
best_rf = clf.fit(X_train, y_train)


#creating rf model and displaying accuracy
print('Random Forest Classification Report:')
rf = RandomForestClassifier(n_estimators=best_rf.best_estimator_.get_params()['n_estimators'])
rf = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))


#random forest confusion matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Random Forest Confusion Matrix', fontsize=18)
plt.show()
