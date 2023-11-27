import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("emails.csv")
print(df)   
# Kopie des Datensatzes
new_df = df.copy()

# Entfernen der ersten und letzten Spalte des Dataframes -> entsprechendes Format
new_df.drop(['Email No.', 'Prediction'], inplace=True, axis=1)

# Erfassen der Häufigkeit der Worte in der Mail
new_df["transformed_text"] = new_df.apply(lambda row: " ".join(sum([[col] * row[col] for col in new_df.columns], [])), axis=1)
df['transformed_text'] = new_df['transformed_text']

# Plot a wordcloud for spam
wordcloud = WordCloud(background_color='white', collocations=False).generate(df[df['Prediction'] == 1]['transformed_text'].str.cat(sep = " "))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Plot a wordcloud for ham
wordcloud = WordCloud(background_color='white', collocations=False).generate(df[df['Prediction'] == 0]['transformed_text'].str.cat(sep = " "))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# #Print dimensionality of the DataFrame
# print(df.shape)

# #Print first 10 rows
# print(df.head(10))

# #Print last 10 rows
# print(df.tail(10))

