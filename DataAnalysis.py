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

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(''.join(df.v2))

plt.figure(figsize=(20, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()