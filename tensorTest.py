import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

xTrain = np.load("X_train.npy")
yTrain = np.load("Y_train.npy")
xTest = np.load("X_test.npy")
yTest = np.load("Y_test.npy")

model = tf.keras.models.Sequential([
  tf.keras.Input(shape=(63,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation='softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics=['accuracy'])

history = model.fit(xTrain, yTrain, epochs=45, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(xTest, yTest, verbose=1)
print(f"\n🎉 FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
Yprediction = model.predict(xTest)
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
plt.savefig("Training graphs.png")
plt.show()
model.save(f"SAI{test_accuracy}.keras")