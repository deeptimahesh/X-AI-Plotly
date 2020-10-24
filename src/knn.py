import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from regression import data, X_train_std, y_train, X_test_std

def run_knn():
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train_std, y_train)

    knn_pred = knn.predict(X_test_std)
    knn_prob = knn.predict_proba(X_test_std)[:,1]
    return knn_prob
