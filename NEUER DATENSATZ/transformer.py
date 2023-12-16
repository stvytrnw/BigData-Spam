import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import pandas as pd
import re
import nltk
from collections import Counter

# dataset = pd.read_csv("./NEUER DATENSATZ/neu.csv")

# dataset["Message"]
# # print(dataset["Message"][1])
# dataset["Message"].fillna("", inplace=True)

# tokenized_messages = [message.split() for message in dataset["Message"]]

# words = [word for message in dataset["Message"] for word in message.split()]

# removed_duplicates = list(dict.fromkeys(words))

# # print(len(words))
# # print(len(removed_duplicates))

# transformed_df = pd.DataFrame(columns=removed_duplicates)
# # transformed_df = pd.DataFrame(columns=removed_duplicates)
# # print(transformed_df)
# print("start loop")
# for i in range(0, len(dataset.index)):
#     word_count = Counter(tokenized_messages[i])
#     transformed_df = pd.concat([transformed_df, pd.DataFrame([word_count])], ignore_index=True)
#     print(i, "/", len(dataset.index))
# print("end loop")
# # for i in range(0, len(dataset.index)):
# #     word_count = {word: tokenized_messages.count(word) for word in transformed_df.columns}
# #     transformed_df = pd.concat([transformed_df, pd.DataFrame([word_count])], ignore_index=True)


# transformed_df.fillna(0, inplace=True)
# # word_count = {word: word_list.count(word) for word in data.columns}
# # data = pd.concat([data, pd.DataFrame([word_count])], ignore_index=True)

# # print(tokenized_messages)
# # print(words)
# transformed_df.to_csv("./NEUER DATENSATZ/transformed3.csv")
# print(transformed_df)

# Liest die CSV-Datei 'neu.csv' aus dem Verzeichnis './NEUER DATENSATZ/' ein und speichert sie in der Variable 'dataset'
dataset = pd.read_csv("./NEUER DATENSATZ/neu.csv")

# Ersetzt alle fehlenden Werte (NaN) in der Spalte 'Message' des Datensatzes durch leere Zeichenketten ("")
dataset["Message"].fillna("", inplace=True)

# Teilt jede Nachricht in der Spalte 'Message' des Datensatzes in eine Liste von Wörtern auf und speichert diese in 'tokenized_messages'
tokenized_messages = dataset["Message"].apply(lambda x: x.split())

# Zählt die Häufigkeit jedes Wortes in jeder Nachricht und speichert diese Zählungen in 'word_counts_list'
word_counts_list = [Counter(message) for message in tokenized_messages]

# Erstellt einen neuen DataFrame 'transformed_df' aus der 'word_counts_list', ersetzt fehlende Werte durch 0 und konvertiert alle Werte in Ganzzahlen
transformed_df = pd.DataFrame(word_counts_list).fillna(0).astype(int)

# Fügt eine neue Spalte 'Email No.' am Anfang des DataFrame 'transformed_df' ein, die die Nummer jeder E-Mail enthält
transformed_df.insert(0, "Email No.", [f"Email {i+1}" for i in range(len(dataset))])

# Fügt eine neue Spalte 'Prediction' am Ende des DataFrame 'transformed_df' ein, die die Kategorie jeder E-Mail aus dem ursprünglichen Datensatz enthält
transformed_df.insert(len(transformed_df.columns), "Prediction", [dataset["Category"][i] for i in range(len(dataset))])

# Speichert den transformierten DataFrame 'transformed_df' als CSV-Datei 'emails2.csv' im Verzeichnis './NEUER DATENSATZ/'
transformed_df.to_csv("./NEUER DATENSATZ/emails2.csv", index=False)

# Gibt den transformierten DataFrame 'transformed_df' auf der Konsole aus
print(transformed_df)

