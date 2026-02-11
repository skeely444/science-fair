import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

xTrain = np.load("X_train.npy")
yTrain = np.load("Y_train.npy")
xTest = np.load("X_test.npy")
yTest = np.load("Y_test.npy")


def add_noise(data, noise_level=0.05):
    # Create a copy so we don't overwrite the original xTest
    noisy_data = np.copy(data)
    # Generate random mask
    jitter = np.random.uniform(0, 1, noisy_data.shape)
    # Add salt and pepper (black and white pixels)
    noisy_data[jitter < (noise_level/2)] = 0    # Pepper
    noisy_data[jitter > (1 - noise_level/2)] = 1 # Salt
    return noisy_data

def change_lighting(data, factor=0.5):
    # factor < 1.0 makes it darker; factor > 1.0 makes it brighter
    dark_data = data * factor
    # Ensure values stay within [0, 1] range
    return np.clip(dark_data, 0, 1)

from scipy.ndimage import gaussian_filter

def apply_blur(data, sigma=1.0):
    blurred_data = np.zeros_like(data)
    for i in range(len(data)):
        # Apply blur to each image in the array
        blurred_data[i] = gaussian_filter(data[i], sigma=sigma)
    return blurred_data

xTest_blurred = apply_blur(xTest, sigma=1.5)
xTest_dark = change_lighting(xTest, factor=0.4)
xTest_noisy = add_noise(xTest, noise_level=0.1) # 10% noise

# Create a 'darkened' version of your training data
xTrain_dark = xTrain * 0.5 
# Create a 'noisy' version
xTrain_noisy = xTrain + np.random.normal(0, 0.05, xTrain.shape)

# Combine them all together
def create_blurred_data(data, window_size=3):
    # This simulates "Blur" for 1D coordinate data
    blurred = np.zeros_like(data)
    for i in range(len(data)):
        # Apply a simple smoothing filter across the 63 features
        blurred[i] = np.convolve(data[i], np.ones(window_size)/window_size, mode='same')
    return blurred

# Create the blurred training examples
xTrain_blurred = create_blurred_data(xTrain)

# Add it to your combined training set
xTrain_combined = np.concatenate([xTrain, xTrain_noisy, xTrain_dark, xTrain_blurred], axis=0)
yTrain_combined = np.concatenate([yTrain, yTrain, yTrain, yTrain], axis=0)

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(63,)),
  tf.keras.layers.GaussianNoise(0.1),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics=['accuracy'])

history = model.fit(xTrain_combined, yTrain_combined, epochs=47, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(xTest, yTest, verbose=1)
print(f"\n🎉 FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_noisy, yTest, verbose=1)
print(f"\n🎉 NOISY TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_dark, yTest, verbose=1)
print(f"\n🎉 BRIGHTNESS TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_blurred, yTest, verbose=1)
print(f"\n🎉 BLURRY TEST ACCURACY: {test_accuracy * 100:.2f}%")

Yprediction = model.predict(xTest)
labledYPred = np.argmax(Yprediction, axis=1)
trueY_labels = yTest
ct = tf.math.confusion_matrix(labels=trueY_labels, predictions=labledYPred)
print(ct.numpy())

Yprediction = model.predict(xTest_noisy)
labledYPred = np.argmax(Yprediction, axis=1)
trueY_labels = yTest
ct = tf.math.confusion_matrix(labels=trueY_labels, predictions=labledYPred)
print(ct.numpy())

Yprediction = model.predict(xTest_dark)
labledYPred = np.argmax(Yprediction, axis=1)
trueY_labels = yTest
ct = tf.math.confusion_matrix(labels=trueY_labels, predictions=labledYPred)
print(ct.numpy())

Yprediction = model.predict(xTest_blurred)
labledYPred = np.argmax(Yprediction, axis=1)
trueY_labels = yTest
ct = tf.math.confusion_matrix(labels=trueY_labels, predictions=labledYPred)
print(ct.numpy())

plt.style.use("ggplot")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="validation accuracy")
plt.title("Accuracy over epoches")
plt.xlabel("Epoches")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training loss")
plt.plot(history.history['val_loss'], label="Validation loss")
plt.title("Loss over epoches")
plt.xlabel("Epoches")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(f"Training graphsFull{test_accuracy}.png")
plt.show()
