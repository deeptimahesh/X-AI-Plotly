import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from regression import data, X_train_std, y_train, X_test_std

# param_grid = { 
#     'n_estimators': [100, 500, 800, 1100],
#     'max_depth' : [10,20,30,50],
# #    'max_features': ['auto', 'sqrt', 'log2'],
#     'criterion' :['gini', 'entropy']
# }
# print("running")
# CV_rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=1), param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train_std, y_train)
# print(CV_rfc.best_params_)

# TAKES SO FKN LONG
## Result from previous commands : n_Estimator = 500, max_depth = 20, criterion = 'entropy'

def run_rf(): 
    rf = RandomForestClassifier( n_estimators=500, max_depth = 33, criterion = 'entropy', random_state = 0).fit(X_train_std, y_train)
    # accuracy = accuracy_score(y_test, rf.predict(X_test_std))
    # 0.848
    y_pred_rf = rf.predict(X_test_std) 
    y_prob_rf = rf.predict_proba(X_test_std)[:,1]
    return y_prob_rf

def run_dt():
    dt = tree.DecisionTreeClassifier(criterion='gini')
    dt = dt.fit(X_train_std, y_train)
    train_pred = dt.predict(X_train_std)
    test_pred = dt.predict(X_test_std)
    dt_prob = dt.predict_proba(X_test_std)[:,1]
    return dt_prob

# print("Accuracy: {0:.3f}".format(accuracy_score(y_test, test_pred)),"\n")
# 0.781 