'''
Machine Learning
Superivsed Learning - Classification
'''

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

bc = load_breast_cancer()
df = pd.DataFrame(bc.data,columns=bc.feature_names)
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size = 0.2, random_state=7)

models = [
          ('LR', LogisticRegression(max_iter=5000)), 
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()), 
          ('GNB', GaussianNB()),
          ('LDA', LinearDiscriminantAnalysis())
        ]

results,names = [], []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=7)
    cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

final = dict(zip(names,results))
for model in list(final.keys()):
    for metric in list(final[model].keys()):
        final[model][metric] = final[model][metric].mean()
        
        
final_df = pd.DataFrame(final).T       
print(final_df)
