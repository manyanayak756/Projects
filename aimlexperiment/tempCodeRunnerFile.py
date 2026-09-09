import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5)

# Test model
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)

# Predict digit
prediction = model.predict(x_test[:1])
print("Predicted Digit:", prediction.argmax())
print("Actual Digit:", y_test[0])

# Show image
plt.imshow(x_test[0], cmap='gray')
plt.show()