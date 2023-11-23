import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("emails.csv")

#Print dimensionality of the DataFrame
print(df.shape)

#Print first 10 rows
print(df.head(10))

#Print last 10 rows
print(df.tail(10))

df = df[df.columns[1:]]

print(' '.join(df[df['Prediction'] == 0]))

#Plot a wordcloud for ham
wordcloud = WordCloud(background_color='white').generate(' '.join(df[df['Prediction'] == 0]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Plot a wordcloud for spam
wordcloud = WordCloud(background_color='white').generate(' '.join(df[df['Prediction'] == 1]))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

