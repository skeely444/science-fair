import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xTrain = np.load("Xnew_train.npy")
xTest = np.load("Xnew_test.npy")
yTrain = np.load("yNew_train.npy")
yTest = np.load("yNew_test.npy")

def change_lightning(data, factor=0.5):
    dark_data = data * factor
    return np.clip(dark_data, 0, 1)

from scipy.ndimage import gaussian_filter

def apply_gaussian_blur(data, sigma=1.0):
    blurred_data = np.zeros_like(data)
    for i in range(len(data)):
        blurred_data[i] = gaussian_filter(data[i], sigma=sigma)
    return blurred_data

def apply_gaussian_noise(data, sigma=0.05):
    noise = np.random.normal(0, sigma, data.shape)
    return np.clip(data + noise, 0, 1)

xTrain_noisyLow = apply_gaussian_noise(xTrain, sigma=0.1)
xTrain_noisyMed = apply_gaussian_noise(xTrain, sigma=0.13)
xTrain_noisyHigh = apply_gaussian_noise(xTrain, sigma=0.2)
xTest_noisy = apply_gaussian_noise(xTest, sigma=0.15)
xTrain_dark = change_lightning(xTrain, factor=0.6)
xTest_dark = change_lightning(xTest, factor = 0.7)
xTrain_blurredLow = apply_gaussian_blur(xTrain, sigma=1.5)
xTrain_blurredMed = apply_gaussian_blur(xTrain, sigma=2.0)
xTrain_blurredHeavy  =apply_gaussian_blur(xTrain, sigma=2.5)
xTest_blurred = apply_gaussian_blur(xTest, sigma=2.3)

xTrain_combined = np.concatenate([
    xTrain, 
    xTrain_noisyLow,
    xTrain_noisyMed,
    xTrain_noisyHigh,
    xTrain_dark, 
    xTrain_blurredLow, 
    xTrain_blurredMed, 
    xTrain_blurredHeavy
], axis=0)
yTrain_combined = np.concatenate([yTrain, yTrain, yTrain, yTrain, yTrain, yTrain, yTrain, yTrain], axis=0)
indices = np.arange(xTrain_combined.shape[0])
np.random.shuffle(indices)
xTrain_combined = xTrain_combined[indices]
yTrain_combined = yTrain_combined[indices]

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(30, 63)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.GaussianNoise(0.1), # Much lower!
  tf.keras.layers.LSTM(300, return_sequences=True, activation='relu'),
  tf.keras.layers.BatchNormalization(), # Added for stability
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.LSTM(200, return_sequences=False,activation='relu'),
  tf.keras.layers.BatchNormalization(), # Added for stability
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(5, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.95)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(xTrain_combined, yTrain_combined, epochs=70, batch_size=32, validation_split=0.2, shuffle=True)
test_loss, test_accuracy = model.evaluate(xTest, yTest, verbose=2)
print(f"\n🎉 REGULAR TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_noisy, yTest, verbose=2)
print(f"\n🎉 NOISY TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_dark, yTest, verbose=2)
print(f"\n🎉 BRIGHTNESS TEST ACCURACY: {test_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(xTest_blurred, yTest, verbose=2)
print(f"\n🎉 BLURRY TEST ACCURACY: {test_accuracy * 100:.2f}% \n \n")

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
plt.savefig(f"TestOne.png")
plt.show()

model.save("TestOne.keras")