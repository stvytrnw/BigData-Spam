import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load your dataset (replace "emails.csv" with your actual file)
df = pd.read_csv("emails.csv")

# Assume you have binary labels (0 or 1) in the last column
X = df[df.columns[1:-1]].values
y = df[df.columns[-1]].values


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define and compile the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
              loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(X_train, y_train, epochs=1000, batch_size=128,
                    validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on the entire test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test Accuracy:', test_accuracy)

# Plotting train and test errors over epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)
plt.title('Training and Test Loss over Epochs')
plt.show()
