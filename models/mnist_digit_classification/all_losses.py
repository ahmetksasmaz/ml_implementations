import numpy as np
import matplotlib.pyplot as plt

i = 1
while True:
    try:
        with open(f'loss_{i}.txt', 'r') as f:
            lines = f.readlines()
            train_losses = []
            test_losses = []
            training_reading = True
            for line in lines:
                if line.startswith('Train Losses:'):
                    continue
                elif line.startswith('Test Losses:'):
                    training_reading = False
                    continue
                if training_reading:
                    train_losses.append(float(line.strip()))
                else:
                    test_losses.append(float(line.strip()))
            plt.plot(train_losses, label='Train Loss '+str(i))
            plt.plot(test_losses, label='Test Loss '+str(i))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
    except FileNotFoundError:
        break
    i += 1
plt.show()