import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


data = pd.read_csv("Open_Univ_Data_Final_merged.csv")
# data.head()
# data.drop(data[data.final_result == 'Withdrawn'].index, inplace=True)
data = data.replace(to_replace ="Distinction", value ="Pass")
data = data.replace(to_replace ="Withdrawn", value ="Fail")
# inst = data.loc[data['id_student']==517269].index.values[0]
# inst1 = data.loc[inst]
# print(inst)

data.set_index('id_student',inplace=True)
# print(data)
data.drop(columns=['date_unregistration', 'Exam_score','TMA_score','CMA_score','mean_score','registration_before','unique' ], inplace=True)
# print(len(data))
data = data.dropna()        # WHAT TO DO ABOUT THIS
# print(data)
train_cols = data.columns[0:-1]
label = data.columns[-1]

X = data[train_cols]
# print(X.loc[inst])
y = data[label].apply(lambda x: 1 if x == "Pass" else 0) #Turning response into 0 and 1

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# print(len(y_train))

X_enc = pd.get_dummies(X, prefix_sep='.')
# print(X_enc.loc[inst])
# feature_names = list(X_enc.columns)
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=0.20, random_state=seed)

sc = StandardScaler()
sc.fit(X_train_enc)
X_train_std = sc.transform(X_train_enc)
X_test_std = sc.transform(X_test_enc)

#LOGISTIC REGRESSION
LR = LogisticRegression(max_iter=10000)
LR.fit(X_train_std, y_train)
# LR.score(X_test_std, y_test)

y_predict = LR.predict(X_test_std)
y_predict_probabilities = LR.predict_proba(X_test_std)[:,1]
# print(len(y_predict_probabilities))
# print(confusion_matrix(y_test, y_predict))
# print(accuracy_score(y_test, y_predict))
# print(f1_score(y_test, y_predict))

# For Probability sakes
sc.fit(X_enc)
X_Pro = sc.transform(X_enc)
Y_Pro = LR.predict_proba(X_Pro)[:,1]
# print(len(Y_Pro))
data['preds'] = Y_Pro
# print((data['preds'].loc[655484]).tolist()[1])


#LOOK UP ACCURACY, OTHER MODELS, MORE GRAPHS, MORE DETAILED

#NOTES ON SOME STUDENT IDS
#678612 : FAIL
#649194: DISTINCTION
#596513: PASS
#279747: DISTINCTION
#655484: WITHDRAWN 