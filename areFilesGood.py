import numpy as np
import os
from sklearn.model_selection import train_test_split

signs = ['good', 'hello', 'okay', 'peace', 'thanks']
X = []  # All landmark data
y = []  # All labels

# Load all data
for sign_id, sign in enumerate(signs):
    folder = f'data/{sign}'
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    for file in files:
        try:
            landmarks = np.load(os.path.join(folder, file))
            if len(landmarks) == 63:  # Valid sample
                X.append(landmarks)
                y.append(sign_id)
        except:
            print(f"Skipping corrupted file: {file}")

X = np.array(X)
y = np.array(y)

print(f"\nTotal samples: {len(X)}")
print(f"Thumbs up: {sum(y==0)}")
print(f"Peace: {sum(y==1)}")
print(f"OK: {sum(y==2)}")

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save for training
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\n✅ Data prepared for training!")