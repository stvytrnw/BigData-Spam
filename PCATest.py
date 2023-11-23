import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.manifold import LocallyLinearEmbedding

df = pd.read_csv("emails.csv")

X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

model = PCA(n_components=0.99)
model.fit(X)
X = model.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# print('kNN')
# model = KNeighborsClassifier(n_neighbors=5)
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('Naive Bayes')
# model = GaussianNB()
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

print('Log Regression')
model = LogisticRegression(max_iter=1000)
print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print('Scores: {}'.format(scores))
print('Mean score: {}'.format(scores.mean()))
print('Std score: {}'.format(scores.std()))
print()

# print('Random Forest Regressor')
# model = RandomForestRegressor()
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('SVM')
# model = svm.SVC(kernel='linear')
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# print('NEURAL NETWORK')
# model = MLPClassifier(random_state=42, max_iter=10000, hidden_layer_sizes=[1000,1000])
# print('Train R2: {}'.format(r2_score(y_train, model.fit(X_train, y_train).predict(X_train))))
# print('Test R2: {}'.format(r2_score(y_test, model.fit(X_train, y_train).predict(X_test))))
# scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print('Scores: {}'.format(scores))
# print('Mean score: {}'.format(scores.mean()))
# print('Std score: {}'.format(scores.std()))
# print()

# Define and compile the neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(100, activation='relu', input_shape=(X.shape[1],)),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(100, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])

# # Perform cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# def plot_history(history, fold):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#     ax1.plot(history.history['loss'], label='Training Loss')
#     ax1.plot(history.history['val_loss'], label='Validation Loss')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Binary Crossentropy')
#     ax1.legend()
#     ax1.grid(True)

#     ax2.plot(history.history['accuracy'], label='Training Accuracy')
#     ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy')
#     ax2.legend()
#     ax2.grid(True)

#     fig.suptitle(f'Fold {fold + 1}')
#     plt.show()

# for fold, (train_index, val_index) in enumerate(cv.split(X, y)):
#     X_train, X_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]

#     # Train the model
#     history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), verbose=0)

#     # Plot the training history for each fold
#     plot_history(history, fold)

# # Evaluate the model on the entire test set
# test_preds = model.predict(X)
# test_preds_binary = np.round(test_preds).flatten()
# test_accuracy = accuracy_score(y, test_preds_binary)
# print('Test Accuracy:', test_accuracy)
