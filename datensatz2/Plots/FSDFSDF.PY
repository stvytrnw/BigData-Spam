import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Laden der Daten aus der CSV-Datei 'emails2.csv' in einen DataFrame 'df'
df = pd.read_csv("emails2.csv")

# Nimmt alle Spalten außer der ersten und der letzten (Email No. & Prediction) aus dem DataFrame und speichert ihre Werte in X
X = df[df.columns[1:-1]].values

# Nimmt die letzte Spalte aus dem DataFrame und speichert ihre Werte in y
y = df[df.columns[-1]].values

# Erstellen eines neuen DataFrame 'X_df' mit den Feature-Daten (alle Spalten außer der ersten und letzten)  
X_df = df[df.columns[1:-1]]

# Dimensionalität des DataFrame ausgeben
print(df.shape)


# Die ersten 10 Zeilen ausgeben
print(df.head(10))

print(df.tail(10))