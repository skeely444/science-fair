import numpy as np

# Load data
X_train = np.load('Xnew_train.npy')
y_train = np.load('yNew_train.npy')
X_test = np.load('Xnew_test.npy')
y_test = np.load('yNew_test.npy')

# Print info
print("="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Feature shape: {X_train.shape}")
print(f"Number of signs: {len(set(y_train))}")

# Count each sign
signs = ['thumbs_up', 'peace', 'ok', 'hello', 'thank_you']
print("\nSamples per sign (training):")
for i, sign in enumerate(signs):
    count = sum(y_train == i)
    print(f"  {sign}: {count}")

print("\nData looks good? Continue to training!")
