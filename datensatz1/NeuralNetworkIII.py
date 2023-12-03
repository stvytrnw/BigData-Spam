import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have already imported necessary libraries

# Load your dataset (replace "emails.csv" with your actual file)
df = pd.read_csv("emails.csv")

# Assume you have binary labels (0 or 1) in the last column
X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values

# Standardize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define and compile the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def plot_history(history, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f'Fold {fold + 1}')
    plt.show()

for fold, (train_index, val_index) in enumerate(cv.split(X, y)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), verbose=0)

    # Plot the training history for each fold
    plot_history(history, fold)

# Evaluate the model on the entire test set
test_preds = model.predict(X)
test_preds_binary = np.round(test_preds).flatten()
test_accuracy = accuracy_score(y, test_preds_binary)
print('Test Accuracy:', test_accuracy)
