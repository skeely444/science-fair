import numpy as np

def add_noise(data, noise_level=0.05):
    # Create a copy so we don't overwrite the original xTest
    noisy_data = np.copy(data)
    # Generate random mask
    jitter = np.random.uniform(0, 1, noisy_data.shape)
    # Add salt and pepper (black and white pixels)
    noisy_data[jitter < (noise_level/2)] = 0    # Pepper
    noisy_data[jitter > (1 - noise_level/2)] = 1 # Salt
    return noisy_data

xTest_noisy = add_noise(xTest, noise_level=0.1) # 10% noise