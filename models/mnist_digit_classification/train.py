import os
import sys
import numpy as np
import cv2 as cv
import torch
import argparse

from DataLoader import DataLoader
from Model import MNISTModel

class Trainer:
    def __init__(self, model, train_loader, test_loader, batch_size, num_epochs, learning_rate, device = "cpu"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Define the loss function and optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in self.test_loader:
                # Move image and label to the device
                image = torch.tensor(image)

                image = image.reshape(-1, 1, 28, 28)

                image = image.to(self.device)

                # Forward pass
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(torch.device("cpu")).item()

                total += 1
                correct += int(predicted == label)

        print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")

    def train(self):
        for epoch in range(self.num_epochs):

            # Get data as batches from the train_loader
            self.train_loader.shuffle_data()
            # Get the batches
            batches = self.train_loader.get_batches(self.batch_size)
            for i in range(0, len(self.train_loader), self.batch_size):
                # Get the images and labels
                batch = next(batches)
                # Unzip the batch into images and labels
                images, labels = zip(*batch)
                images = torch.tensor(images)
                labels_array = np.zeros((batch_size, 10), dtype=np.float32)
                for j, label in enumerate(labels):
                    # Convert the label to one-hot encoding
                    labels_array[j][label] = 1
                labels = torch.tensor(np.array(labels_array), dtype=torch.float32)

                images = images.reshape(-1, 1, 28, 28)
                labels = labels.reshape(-1, 10)

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                output = self.model(images)

                # Compute the loss
                loss = self.criterion(output, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                # loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}")
            self.test()

if __name__ == "__main__":
    args = sys.argv

    parser = argparse.ArgumentParser(description="MNIST Digit Classification")
    parser.add_argument("--train_path", type=str, default="mnist_data_train.csv", help="Path to the MNIST dataset")
    parser.add_argument("--test_path", type=str, default="mnist_data_test.csv", help="Path to the MNIST dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--model_path", type=str, default="mnist_model.pth", help="Path to save the trained model")
    parser.add_argument("--use_mps", action="store_true", help="Use MPS for training")

    # Parse the command line arguments
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    model_path = args.model_path
    use_mps = args.use_mps

    # Check if MPS is available

    device = None
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    elif not use_mps:
        device = torch.device("cpu")
        print("Using CPU for training.")
    else:
        print("MPS device not found.")
        exit(1)
    
    # Load the MNIST dataset
    train_loader = DataLoader(train_path)
    test_loader = DataLoader(test_path)
    train_loader.load_data()
    test_loader.load_data()
    train_loader.shuffle_data()

    model = MNISTModel().to(device)

    trainer = Trainer(model, train_loader, test_loader, batch_size, num_epochs, learning_rate, device)

    trainer.train()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")