# -*- coding: utf-8 -*-
"""
How Many Bikes ?
Kaggle Challenge user name: alextuil

Alexis Tuil
12/16/2016
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import make_scorer

def RMSLE(y, y_pred):
    y_pred[y_pred < 0] = 0
    return np.sqrt( np.mean((np.log(y_pred + 1) - np.log(y + 1))**2) )

dataset = pd.read_csv('train.csv', header=0)
y, X = dmatrices('cnt ~ 0 + yr + season + mnth + hr + holiday + weekday + workingday + weathersit + temp + atemp + hum + windspeed',dataset, return_type='dataframe')


# set up a k-fold cross validation on the training set

kf = KFold(n_splits=5)


# linear regression on the full training data

lr = LinearRegression()
lr.fit(X,y)
y_pred = lr.predict(X)

print(RMSLE(y,y_pred))
# 1.300951

# linear regression cross-validated

scores = []

y = np.ravel(y)
Xm = X.as_matrix()
for train, test in kf.split(Xm):
    model = LinearRegression()
    model.fit(Xm[train],y[train])
    y_pred = model.predict(Xm[test])
    scores.append(RMSLE(y[test],y_pred))

print(np.mean(scores))
# 1.36903506661 

# ridge regression on the full training data

a_range = np.arange(0,3000,10)
a_score = []
for a in a_range:
    rdg = Ridge(alpha=a)
    rdg.fit(X,y)
    y_pred = rdg.predict(X)
    a_score.append(RMSLE(y,y_pred))
    
plt.plot(a_range, a_score)
plt.xlabel('Ridge Parameter')
plt.ylabel('RMSLE score')
plt.title('Ridge regression on the full training set')

# best for alpha=750
rdg = Ridge(alpha=750)
rdg.fit(X,y)
print(RMSLE(y,rdg.predict(X)))
# RMSLE = 1.23244788655

# c) ridge regression cross-validated
a_range = np.arange(0,2000,10)
a_score = []

for a in a_range: 
    
    scores = []

    for train, test in kf.split(Xm):
        rdg = Ridge(alpha=a)
        rdg.fit(Xm[train],y[train])
        y_pred = rdg.predict(Xm[test])
        
        scores.append(RMSLE(y[test],y_pred))
    a_score.append(np.mean(scores))

plt.plot(a_range, a_score)
plt.xlabel('Ridge Parameter')
plt.ylabel('RMSLE score')
plt.title('Ridge regression cross-validated')

# best for alpha=500
scores = []
for train, test in kf.split(Xm):
    rdg = Ridge(alpha=500)
    rdg.fit(Xm[train],y[train])
    scores.append(RMSLE(y[test],rdg.predict(Xm[test])))
print(np.mean(scores))
# RMSLE = 1.24320579537

# ridge regression on the full training data with scaling

X_scaled = scale(X)

# when training then evaluating on the full training set
a_range = np.arange(0,6000,10)
a_score = []
for a in a_range:
    rdg = Ridge(alpha=a)
    rdg.fit(X_scaled,y)
    y_pred = rdg.predict(X_scaled)
    a_score.append(RMSLE(y,y_pred))
  
plt.plot(a_range, a_score)
plt.xlabel('Ridge Parameter')
plt.ylabel('RMSLE score')
plt.title('Ridge regression on the full training set and features scaled')

# best for alpha=3500
rdg = Ridge(alpha=3500)
rdg.fit(X_scaled,y)
print(RMSLE(y,rdg.predict(X_scaled)))
# RMSLE = 1.26469309761

# ridge regression cross-validated with scaling

a_range = np.arange(0,6000,1)
a_score = []

for a in a_range: 
    
    scores = []

    for train, test in kf.split(Xm):
        rdg = Ridge(alpha=a)
        
        scaler = StandardScaler().fit(Xm[train])
        Xm_train_scaled = scaler.transform(Xm[train]) 
        rdg.fit(Xm_train_scaled,y[train])
        Xm_test_scaled = scaler.transform(Xm[test])
        
        y_pred = rdg.predict(Xm_test_scaled)
        
        scores.append(RMSLE(y[test],y_pred))
    a_score.append(np.mean(scores))
    
plt.plot(a_range, a_score)
plt.xlabel('Ridge Parameter')
plt.ylabel('RMSLE score')
plt.title('Ridge regression cross-validated and features scaled')

# best for alpha = 3500
scores = []
for train, test in kf.split(Xm):
    rdg = Ridge(alpha=3500)
        
    scaler = StandardScaler().fit(Xm[train])
    Xm_train_scaled = scaler.transform(Xm[train]) 
    rdg.fit(Xm_train_scaled,y[train])
    Xm_test_scaled = scaler.transform(Xm[test])
        
    y_pred = rdg.predict(Xm_test_scaled)
        
    scores.append(RMSLE(y[test],y_pred))
print(np.mean(scores))
# RMSLE = 1.27989927838



dataset_test = pd.read_csv('test.csv', header=0)
X_test = dataset_test[['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']]
model = linear_model.Ridge(alpha=500)
model.fit(X,y)
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i] = math.floor(max(0,y_pred[i]))

d = {'id': np.ravel(dataset_test[['instant']]), 'prediction': y_pred}
df = pd.DataFrame(d,)
df.to_csv("submission-1.csv", index=False)

# sent to Kaggle Leaderboard RMLSE = 1.29616  


RMSLE_scoring = make_scorer(score_func=RMSLE, greater_is_better=False)

# (a) Linear kernel

c_range = np.logspace(-6, 5, 9)
scores = []

for c in c_range:
    
    rmlse_scores = []
    model = svm.LinearSVR(C=c)
    score = - cross_val_score(model, X, y, cv=5, scoring=RMSLE_scoring)
    scores.append(np.mean(score))

plt.semilogx(c_range, scores)
plt.xlabel('C')
plt.ylabel('RMLSE score')
plt.title('Linear kernel')
plt.show()

# Gaussian RDB kernel

c_range = np.logspace(0, 2, 3)
gamma_range = np.logspace(-3, 0, 4)
scores = pd.DataFrame()

for c in c_range:
    for g in gamma_range:
    
        rmlse_scores = []
        model = svm.SVR(kernel='rbf', gamma=g, C=c)
        score = -cross_val_score(model, X, y, cv=5, scoring=RMSLE_scoring)
        res = pd.DataFrame({'C': c, 'gamma': g, 'RMSLE': np.mean(score)}, index=np.arange(1))
        scores = scores.append(res)


# Test Data

model = svm.SVR(kernel='rbf', gamma=0.1, C=100)
model.fit(X,y)
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i] = math.floor(max(0,y_pred[i]))
    
d = {'id': np.ravel(dataset_test[['instant']]), 'prediction': y_pred}
df = pd.DataFrame(d,)
df.to_csv("submission-2.csv", index=False)

# sent to Kaggle Leaderboard RMLSE = 0.61700


# Random Forest for different parameters

max_depth_range = [10,15,20,None]
min_samples_leaf_range= [2,5,10,20]
scores = pd.DataFrame()

for md in max_depth_range:
    for msl in min_samples_leaf_range:
    
        rmlse_scores = []
        model = RandomForestRegressor(n_estimators=500, max_depth=md, min_samples_leaf=msl, max_features='auto', n_jobs=-1, random_state=0)
        score = -cross_val_score(model, X, y, cv=5, scoring=RMSLE_scoring)
        res = pd.DataFrame({'Max Depth': md, 'Min Sample Leaf': msl, 'RMSLE': np.mean(score)}, index=np.arange(1))
        scores = scores.append(res)

print(scores)

# best for max_depth = None and min_sample_leaf = 2

model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, max_features='auto', n_jobs=-1, random_state=0)
score = - cross_val_score(model, X, y, cv=5, scoring=RMSLE_scoring)
print(np.mean(score))
# RMLSE = 0.470432421612

# Test Data
model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2, max_features='auto', n_jobs=-1, random_state=0)
model.fit(X,y)
y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    y_pred[i] = math.floor(max(0,y_pred[i]))
    
d = {'id': np.ravel(dataset_test[['instant']]), 'prediction': y_pred}
df = pd.DataFrame(d,)
df.to_csv("submission-3.csv", index=False)

# sent to Kaggle Leaderboard RMLSE = 0.47329
