import numpy as np
import random
import matplotlib.pyplot as plt

SAMPLE_COUNT = 1000
PARAMETER_COUNT = 100
MAX_PARAMETER = 10.0
MAX_SAMPLE = 100
MAX_NOISE_RATIO = 1/5.0
LEARNING_RATE = 0.0001
MAX_ITERATION = 20

# Create random weights

weights = (np.random.random((PARAMETER_COUNT,1)) * 2.0 - 1.0) * MAX_PARAMETER
bias = random.random() * MAX_PARAMETER

# Create random samples
x = (np.random.random((SAMPLE_COUNT,PARAMETER_COUNT)).astype(np.float32) * 2.0 - 1.0) * MAX_SAMPLE

# Find ground truth measurements of samples
y = np.matmul(x,weights) + bias

# Add noise ratio
noise_ratio = (np.random.random((SAMPLE_COUNT,1)).astype(np.float32) * 2.0 - 1.0) * MAX_NOISE_RATIO

# Add noise to measurements
noisy_y = y + y*noise_ratio

optimized_weights = np.zeros((PARAMETER_COUNT,1))
optimized_bias = 0.0

avg_weight_errors = []
mean_square_errors = []

iteration_step = 0
while iteration_step <= MAX_ITERATION:
    y_hat = np.matmul(x,optimized_weights) + optimized_bias
    y_diff = noisy_y - y_hat
    error = np.mean(np.power(y_diff, 2))

    for i in range(len(optimized_weights)):
        d_error_d_weight = np.mean(-2*np.transpose(y_diff)*x[:,i])
        optimized_weights[i] += -LEARNING_RATE * d_error_d_weight
    
    d_error_d_bias = np.mean(-2*np.transpose(y_diff))
    optimized_bias += -LEARNING_RATE * d_error_d_bias

    avg_weight_errors.append(np.mean(np.abs(weights - optimized_weights)))
    mean_square_errors.append(error)

    iteration_step += 1

plt.ylim(0, np.max(avg_weight_errors))
plt.plot(avg_weight_errors)
plt.show()

plt.ylim(0,np.max(mean_square_errors))
plt.plot(mean_square_errors)
plt.show()