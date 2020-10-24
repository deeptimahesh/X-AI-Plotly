import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Open_Univ_Data_Final_merged.csv")

from xgboost import XGBClassifier, plot_importance
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

df.drop(columns=['date_unregistration', 'Exam_score','TMA_score','CMA_score','mean_score','registration_before','unique', 'age_band' ], inplace=True)
df.set_index('id_student',inplace=True)
df = df.dropna()

target = df.columns[-1]
train_cols = df.columns[0:-1]

features_list = list(df.columns)[0:-1]
df = df.replace(to_replace ="Distinction", value ="Pass")
df = df.replace(to_replace ="Withdrawn", value ="Fail")
data = df
data = data.replace(to_replace ="Pass", value =1)
data = data.replace(to_replace ="Fail", value =0)


X = data[train_cols]
X_enc = pd.get_dummies(X, prefix_sep='.')

y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size = 0.3, random_state = 33)
xgb = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs = -1)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
