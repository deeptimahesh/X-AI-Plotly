import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import eli5
from skll.metrics import spearman

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import warnings
from concurrent.futures import ProcessPoolExecutor

import lime
from lime.lime_tabular import LimeTabularExplainer
import shap

# setting up the styling for the plots in this notebook
sns.set(style="white", palette="colorblind", font_scale=1.2, 
        rc={"figure.figsize":(12,9)})
RANDOM_STATE = 420
N_JOBS=8

print("Running ExAi...")

data = pd.read_csv("Open_Univ_Data_Final_merged.csv")

data.drop(columns=['date_unregistration', 'Exam_score','TMA_score','CMA_score','mean_score','registration_before','unique' ], inplace=True)
data = data.replace(to_replace='55<=', value = '55')
data.head()

data = data.dropna()
data = data.replace(to_replace ="Distinction", value ="Pass")
data = data.replace(to_replace ="Withdrawn", value ="Fail")
data.drop(columns=['repeatactivity','sharedsubpage','url'], inplace=True)
data.drop(columns=['dualpane','glossary','date_registration_pos'], inplace=True)
data.drop(columns=['ouelluminate','subpage','questionnaire'], inplace=True)
data.drop(columns=['htmlactivity','externalquiz','folder','num_of_prev_attempts', 'forumng'], inplace=True)

train_cols = data.columns[0:-1]
label = data.columns[-1]
X = data[train_cols]
y = data[label].apply(lambda x: 1 if x == "Pass" else 0)

seed = 1
X_enc = pd.get_dummies(X, prefix_sep='.')

a = list(X_enc['id_student'])
X_enc.drop(columns=['id_student'], inplace=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_enc)
X_std = sc.transform(X_enc)
X_a = pd.DataFrame(X_std, index=X_enc.index, columns=X_enc.columns)
X_a['id'] = a

X_train, X_test, y_train, y_test = train_test_split(X_a, y, test_size=0.25, random_state=seed)

features = list(X_train.columns)
features.remove('id')

train = X_train[features]

pipe = Pipeline([("imputer", Imputer()),
                 ("estimator", RandomForestClassifier(random_state=RANDOM_STATE))])
spearman_scorer = make_scorer(spearman)
rf_param_space = {
    'imputer__strategy': Categorical(['mean', 'median', 'most_frequent']),
    'estimator__max_features': Integer(1, 8),
    'estimator__n_estimators': Integer(50, 500), 
    'estimator__min_samples_split': Integer(2, 200),
}

search = BayesSearchCV(pipe, 
                      rf_param_space, 
                      cv=10,
                      n_jobs=N_JOBS, 
                      verbose=0, 
                      error_score=-9999, 
                      scoring=spearman_scorer, 
                      random_state=RANDOM_STATE,
                      return_train_score=True, 
                      n_iter=3)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    search.fit(train, y_train) 

test = X_test[features]

predicted = search.predict(test)
model_test_score = spearman_scorer(search, test, y_test)

estimator = search.best_estimator_.named_steps['estimator']
imputer = search.best_estimator_.named_steps['imputer']


def multiproc_iter_func(max_workers, an_iter, func, item_kwarg, **kwargs):
    """
    A helper functions that applies a function to each item in an iterable using
    multiple processes. 'item_kwarg' is the keyword argument for the item in the
    iterable that we pass to the function.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = [executor.submit(func, **{item_kwarg: item}, **kwargs)
                          for item in an_iter]

        results = [future.result() for future in future_results]
        
    return results

train_X_imp = imputer.transform(train.head(480))
test_X_imp = imputer.transform(test.head(480))

train_X_imp_df = pd.DataFrame(train_X_imp, columns=features)
explainer = LimeTabularExplainer(train_X_imp_df, mode='classification', 
                                 feature_names=features, 
                                 random_state=RANDOM_STATE, 
                                 discretize_continuous=False) 
test_X_imp_df = pd.DataFrame(test_X_imp, columns=features)

# print(test_X_imp_df.columns)

test_X_imp_df = pd.DataFrame(test_X_imp, columns=features)
# the number of features to include in our predictions
num_features = len(features)
# the index of the instance we want to explaine
exp_idx = 0
exp = explainer.explain_instance(test_X_imp_df.iloc[exp_idx,:].values, 
                                 estimator.predict_proba, num_features=num_features)

shap_explainer = shap.TreeExplainer(estimator)
# calculate the shapley values for our test set
test_shap_vals = shap_explainer.shap_values(test_X_imp)

# exp = exp.as_list(label=1)
# fig = plt.figure()
# vals = [x[1] for x in exp]
# names = [x[0] for x in exp]
# vals.reverse()
# names.reverse()
# colors = ['green' if x > 0 else 'red' for x in vals]
# pos = np.arange(len(exp)) + .5
# plt.barh(pos, vals, align='center', color=colors)
# plt.yticks(pos, names)
# title = 'Local explanation for class %s' % list(X_test.id)[exp_idx]
# plt.title(title)
# shap.force_plot(explainer.expected_value, shap_values[0,:], 
#     X.iloc[0,:],show=False,matplotlib=True)
#     .savefig('scratch.png')

