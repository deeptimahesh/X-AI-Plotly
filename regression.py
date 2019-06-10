import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix


df = pd.read_csv("Open_Univ_Data_Final_merged.csv")
# df.head()
df.drop(df[df.final_result == 'Withdrawn'].index, inplace=True)
df = df.replace(to_replace ="Distinction", value ="Pass")

df.drop(columns=['id_student','date_unregistration', 'Exam_score','TMA_score','CMA_score','mean_score','registration_before','unique' ], inplace=True)
df = df.dropna()

train_cols = df.columns[0:-1]
label = df.columns[-1]

X = df[train_cols]
y = df[label].apply(lambda x: 1 if x == "Pass" else 0) #Turning response into 0 and 1

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

X_enc = pd.get_dummies(X, prefix_sep='.')
feature_names = list(X_enc.columns)
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=0.20, random_state=seed)

sc = StandardScaler()
sc.fit(X_train_enc)
X_train_std = sc.transform(X_train_enc)
X_test_std = sc.transform(X_test_enc)

#LOGISTIC REGRESSION
LR = LogisticRegression()
LR.fit(X_train_std, y_train)
# LR.score(X_test_std, y_test)

y_predict = LR.predict(X_test_std)
y_predict_probabilities = LR.predict_proba(X_test_std)[:,1]
# print(confusion_matrix(y_test, y_predict))

