import numpy as np
import random

SAMPLE_COUNT = 10000
MAX_SLOPE = 10.0
MAX_OFFSET = 1.0
MAX_SAMPLE = 100
MAX_NOISE_RATIO = 1/100.0
LEARNING_RATE = 0.000000001
MAX_ITERATION = 100

# Create a 2D line with random slope and offset
# y = slope*x + offset

slope = random.random() * MAX_SLOPE
offset = random.random() * MAX_OFFSET


# Create random samples from 2D line
x = (np.random.random((1,SAMPLE_COUNT)).astype(np.float32) * 2.0 - 1.0) * MAX_SAMPLE

# Find ground truth measurements of samples
y = x * slope + offset

# Add noise ratio
noise_ratio = (np.random.random((1,SAMPLE_COUNT)).astype(np.float32) * 2.0 - 1.0) * MAX_NOISE_RATIO

# Add noise to measurements

noisy_y = y + y * noise_ratio

# print(slope)
# print(offset)
# print(x)
# print(y)
# print(noise_ratio)
# print(noisy_y)

# Set initial weights of our model

w_0 = 0.0
w_b = 0.0

# print(w_0)
# print(w_b)

# y_i = x_i * w_0 + w_b
# y^ = x * w_0 + w_b
# error = sum((y - y^)^2)

# Iteration block
iteration_step = 0
while iteration_step <= MAX_ITERATION:
    y_hat = x * w_0 + w_b
    y_diff = y - y_hat
    error = np.sum(np.power(y_diff, 2))

    # print("Current parameters : ", w_0, w_b, " with error : ", error)

    # sum((y - x * w_0 - w_b)^2)
    # d_error / d_w_0 = sum(-2*x_i*(y_diff_i))
    # d_error / d_w_b = sum(-2*(y_diff_i))

    d_error_d_w_0 = np.sum(-2*x*(y_diff))
    d_error_d_w_b = np.sum(-2*(y_diff))

    w_0 += -LEARNING_RATE * d_error_d_w_0
    w_b += -LEARNING_RATE * d_error_d_w_b
    iteration_step += 1

print("Last parameters : ", w_0, w_b, " with error : ", error)
print("Actual parameters : ", slope, offset)
