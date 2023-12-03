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
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Import the Dataframe
df = pd.read_csv("./NEUER DATENSATZ/SPAM text message 20170820 - Data.csv")

le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])


def clean_text(text):
    # Make text lowercase
    text = text.lower()

    # Remove text in square brackets
    text = re.sub('\[.*?\]', '', text)

    # Remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove punctuation
    text = re.sub('[^a-zA-Z0-9\s]+', '', text)

    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)

    # Remove extra whitespace
    text = re.sub('\s+', ' ', text).strip()

    return text


# Apply the clean_text function to all text in the 'text' column
df['Message'] = df['Message'].apply(clean_text)

# initialize SnowballStemmer
stemmer = SnowballStemmer('english')


def stem_text(text):
    # Tokenize the input text into individual words
    tokens = nltk.word_tokenize(text)

    # Stem each token using the SnowballStemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Join the stemmed tokens back into a single string
    return ' '.join(stemmed_tokens)


# apply stemming to the 'text' column in your DataFrame
df['Message'] = df['Message'].apply(stem_text)

# Define TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the text data
X = vectorizer.fit_transform(df['Message'])

# Define target variable
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

score_data = {'algorithms': [],
              'train_scores': [],
              'test_scores': [],
              'cross_val_means': [],
              'cross_val_stds': []}


def getScores(name):
    print(name)
    score_data['algorithms'].append(name)
    train_score = r2_score(y_train, model.fit(
        X_train, y_train).predict(X_train))
    score_data['train_scores'].append(train_score)
    print('Train R2: {}'.format(train_score))
    test_score = r2_score(y_test, model.fit(X_train, y_train).predict(X_test))
    score_data['test_scores'].append(test_score)
    print('Test R2: {}'.format(test_score))
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print('Scores: {}'.format(scores))
    print('Mean score: {}'.format(scores.mean()))
    print('Std score: {}'.format(scores.std()))
    score_data['cross_val_means'].append(scores.mean())
    score_data['cross_val_stds'].append(scores.std())
    print()

# df.insert(0, "Email No.", ["Email {}".format(i) for i in range(1, len(df) + 1)])
# df.to_csv("./NEUER DATENSATZ/neu.csv", index=False)
X.to


# model = KNeighborsClassifier(n_neighbors=5)
# getScores('kNN')

# # model = GaussianNB()
# # getScores('Naive Bayes')

# model = LogisticRegression(max_iter=1000)
# getScores('Log Regression')

# model = RandomForestClassifier()
# getScores('Random Forest Classifier')

# model = svm.SVC(kernel='linear')
# getScores('SVM')

# # model = MLPClassifier(random_state=42, max_iter=10000,
# #                       hidden_layer_sizes=[1000, 1000])
# # getScores('NEURAL NETWORK')

# # Erstellen Sie ein DataFrame mit Ihren Daten
# data = pd.DataFrame(list(zip(score_data['algorithms'], score_data['train_scores'], score_data['test_scores'], score_data['cross_val_means'],
#                     score_data['cross_val_stds'])), columns=['Algorithmen', 'Train R2', 'Test R2', 'Mean', 'Std'])

# # Setzen Sie die Algorithmen als Index
# data.set_index('Algorithmen', inplace=True)

# # Erstellen Sie die Heatmap
# sns.heatmap(data, annot=True, cmap='viridis')
# plt.title('Vergleich der R2-Scores verschiedener Algorithmen')
# plt.show()
