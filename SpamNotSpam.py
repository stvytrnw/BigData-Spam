import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("emails.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# PCA : 

print('kNN')
model = KNeighborsClassifier(n_neighbors=5)
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('Naive Bayes')
model = GaussianNB()
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('Log Regression')
model = LogisticRegression(max_iter=1000)
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('Random Forest Regressor')
model = RandomForestRegressor()
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

print('SVM')
model = svm.SVC(kernel='linear')
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

# print('NEURAL NETWORK')
# model = MLPClassifier(random_state=42, max_iter=10000, hidden_layer_sizes=[100,100])
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()


# Tensor Flow Cross Validation?
print('NEURAL NETWORK')
def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()
  
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(3000,)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy',
                metrics=['accuracy'])
history = model.fit(
  X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=0
)

plot_history(history)
accuracy = model.evaluate(X_test, y_test)[1]
print('Accuracy:', accuracy)
