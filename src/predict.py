import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = keras.models.load_model("../models/mnist_cnn.h5")

# Load the MNIST dataset again (for testing)
(_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
X_test = X_test / 255.0  # Normalize
X_test = X_test.reshape(-1, 28, 28, 1)

# Pick a random image
index = np.random.randint(0, len(X_test))
img = X_test[index]

# Predict
prediction = np.argmax(model.predict(img.reshape(1, 28, 28, 1)))

# Plot the image
plt.imshow(img.reshape(28,28), cmap="gray")
plt.title(f"Predicted Label: {prediction}")
plt.show()
