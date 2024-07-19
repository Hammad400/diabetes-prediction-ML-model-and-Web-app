import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

df = pd.read_csv('kaggle_diabetes.csv')
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

X = df.drop(columns='Outcome')
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

model=classifier
y_predicted=model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_predicted))
print("Precision:",metrics.precision_score(y_test, y_predicted))
print("Recall:",metrics.recall_score(y_test, y_predicted))

score=metrics.accuracy_score(y_test,y_predicted)
pre=metrics.precision_score(y_test, y_predicted)
recall=metrics.recall_score(y_test, y_predicted)
f1_score=2*pre*recall/(pre+recall)
print("f1_score:",f1_score)

filename = 'diabetes-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
print("model is created!")