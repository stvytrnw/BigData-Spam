import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pickle

df = pd.read_csv("../emails.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# print('Random Forest Classifier')
model = RandomForestClassifier()
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# model.fit(X_train, y_train)
# filename = 'model.sav'
# pickle.dump(model, open(filename, 'wb'))
# joblib.dump(value=model, filename="model.joblib")

def transform(mail):
    # Neue Mail als String -> Liste mit einzelnen Wörtern der Mail
    # new_mail = "Hello I need the help"
    word_list = mail.lower().split()
    # print(word_list)

    # Kopie des Datensatzes
    new_df = df.copy()

    # Entfernen der ersten und letzten Spalte des Dataframes -> entsprechendes Format
    new_df.drop(['Email No.', 'Prediction'], inplace=True, axis=1)

    # Neuer Dataframe
    data = pd.DataFrame(columns=new_df.columns)

    # Erfassen der Häufigkeit der Worte in der Mail
    word_count = {word: word_list.count(word) for word in data.columns}
    data = pd.concat([data, pd.DataFrame([word_count])], ignore_index=True)

    # Auffüllen der fehlenden Werte
    data = data.fillna(0)

    return data


mail = transform("Hello my name is Johann")
model = pickle.load(open('model.sav', 'rb'))
model.predict(mail)

