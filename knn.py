import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_std, y_train)

train_accuracy = knn.score(X_train_std, y_train)
test_accuracy = knn.score(X_test_std, y_test)
y_pred = knn.predict(X_test_std)

print(train_accuracy)
print(test_accuracy)
print(f1_score(y_test, y_pred))
