import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("emails2.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values

X_df = df[df.columns[1:-1]]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = PCA(n_components=2)
model.fit(X)
X = model.transform(X)


df_pca = pd.DataFrame(model.components_)

# Sortiere die Spalten nach den Werten in der ersten Reihe
sorted_df = df_pca.sort_values(by=df.index[0], axis=1, ascending=False)

# Behalte nur die ersten 200 Spalten
filtered_df = sorted_df.iloc[:, :200]

selected_columns = [int(col) for col in filtered_df.columns] 

# Erstellen Sie eine Liste der Spaltennamen aus df_large basierend auf den Indizes
selected_columns_names = [X_df.columns[i] for i in selected_columns]

df_large_reduced = X_df.iloc[:, selected_columns]

# Erfassen der Häufigkeit der Worte in der Mail
df_large_reduced["transformed_text"] = df_large_reduced.apply(lambda row: " ".join(
    sum([[col] * row[col] for col in df_large_reduced.columns], [])), axis=1)
df['transformed_text'] = df_large_reduced['transformed_text']

# Plot a wordcloud for spam
wordcloud = WordCloud(background_color='white', collocations=False).generate(
    df[df['Prediction'] == 1]['transformed_text'].str.cat(sep=" "))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Plot a wordcloud for ham
wordcloud = WordCloud(background_color='white', collocations=False).generate(
    df[df['Prediction'] == 0]['transformed_text'].str.cat(sep=" "))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Print dimensionality of the DataFrame
print(df.shape)

# Print first 10 rows
print(df.head(10))

# Print last 10 rows
print(df.tail(10))
