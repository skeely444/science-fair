import tensorflow as tf
import numpy as np
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

model.fit(xTrain, yTrain, epochs=50, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(xTest, yTest, verbose=1)
print(f"\n🎉 FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
model.save("SAI.keras")