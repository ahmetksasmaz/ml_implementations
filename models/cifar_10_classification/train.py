import os
import sys
import numpy as np
import cv2 as cv
import torch
import argparse

from DataLoader import DataLoader
from Model import *
from Plotter import Plotter

class Trainer:
    def __init__(self, model, plotter, train_loader, test_loader, batch_size, num_epochs, learning_rate, device = "cpu"):
        self.model = model
        self.plotter = plotter
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Define the loss function and optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.test_criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        sum_loss = 0.0

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

                # Compute the loss
                labels_array = np.zeros(10, dtype=np.float32)
                labels_array[label] = 1

                labels_array = torch.tensor(labels_array)
                labels_array = labels_array.reshape(-1, 10)
                labels_array = labels_array.to(self.device)
                sum_loss += self.test_criterion(outputs, labels_array).item()

                total += 1
                correct += int(predicted == label)
            plotter.add_loss(sum_loss / len(self.test_loader))

        print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")

    def train(self):
        for epoch in range(self.num_epochs):

            sum_loss = 0.0

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
                sum_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                # loss.requires_grad = True
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}")
            plotter.add_loss(sum_loss / (len(self.train_loader) / float(self.batch_size)), is_train=True)
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
    parser.add_argument("--test_only", action="store_true", help="Test only the model")

    # Parse the command line arguments
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    model_path = args.model_path
    use_mps = args.use_mps
    test_only = args.test_only

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

    models = []

    if not test_only:
        models.append(CIFARModel().to(device))
    else:
        base, ext = os.path.splitext(model_path)
        i = 0
        while True:
            new_model_path = f"{base}_{i+1}{ext}"
            if not os.path.exists(new_model_path):
                break
            model = None
            if i == 0:
                model = CIFARModel().to(device)
            else:
                print(f"Model {i+1} not found.")
                break
            model.load_state_dict(torch.load(new_model_path, map_location=device))
            models.append(model)
            i += 1

    run_only = []

    for index, model in enumerate(models):
        if index+1 not in run_only and len(run_only) > 0:
            continue
        if test_only:
            print(f"Testing model {index + 1}...")
        else:
            print(f"Training model {index + 1}...")
        plotter = Plotter()

        trainer = Trainer(model, plotter, train_loader, test_loader, batch_size, num_epochs, learning_rate, device)

        if test_only:
            trainer.test()
            continue
        
        trainer.train()

        # If model path exists create new model path with sequence number
        base, ext = os.path.splitext(model_path)
        new_model_path = f"{base}_{index+1}{ext}"

        # Save the model
        plotter.plot_loss(index+1)
        plotter.save_loss(index+1)

        torch.save(model.state_dict(), new_model_path)
        print(f"Model saved to {new_model_path}")