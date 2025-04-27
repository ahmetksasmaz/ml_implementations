import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
    
    def add_loss(self, loss, is_train=False):
        if is_train:
            self.train_losses.append(loss)
        else:
            self.test_losses.append(loss)

    def plot_loss(self, file_number):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(f'loss_curve_{file_number}.png')
        plt.close()
    
    def save_loss(self, file_number):
        with open(f'loss_{file_number}.txt', 'w') as f:
            f.write('Train Losses:\n')
            for loss in self.train_losses:
                f.write(f'{loss}\n')
            f.write('Test Losses:\n')
            for loss in self.test_losses:
                f.write(f'{loss}\n')