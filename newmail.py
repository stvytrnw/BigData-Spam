import pandas as pd

df = pd.read_csv("emails.csv")


# Neue Mail als String -> Liste mit einzelnen Wörtern der Mail
new_mail = "Hello I need the help"
word_list = new_mail.lower().split()
# print(word_list)


# Kopie des Datensatzes
new_df = df.copy()


# Entfernen der ersten und letzten Spalte des Dataframes -> entsprechendes Format
new_df.drop(['Email No.', 'Prediction'], inplace=True, axis=1)


# Neuer Dataframe
data = pd.DataFrame(columns=new_df.columns)


# Erfassen der Häufigkeit der Worte in der Mail
word_count = {word: word_list.count(word) for word in data.columns}
data = data.append(word_count, ignore_index=True)


# Auffüllen der fehlenden Werte
data = data.fillna(0)


print(data)