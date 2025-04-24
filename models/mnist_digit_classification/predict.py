import os
import sys
import numpy as np
import cv2 as cv
import torch
import argparse

from DataLoader import DataLoader
from Model import MNISTModel

class Predictor:
    def __init__(self, model, device = "cpu"):
        self.model = model
        self.device = device

    def predict(self, image):
        self.model.eval()

        with torch.no_grad():
                image = torch.tensor(image)
                image = image.reshape(-1, 1, 28, 28)
                image = image.to(self.device)

                # Forward pass
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to(torch.device("cpu")).item()

                print(f"Predicted label: {predicted}")

if __name__ == "__main__":
    args = sys.argv

    parser = argparse.ArgumentParser(description="MNIST Digit Classification Predictor")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the MNIST dataset")
    parser.add_argument("--model_path", type=str, default="mnist_model.pth", help="Path to save the trained model")
    parser.add_argument("--use_mps", action="store_true", help="Use MPS for training")

    # Parse the command line arguments
    args = parser.parse_args()
    image_path = args.image_path
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
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load the image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28, 28))
    image = cv.bitwise_not(image)
    image = image.astype(np.float32) / 255.0
    image = image.reshape(1, 28, 28)
    image = image.astype(np.float32)

    # Create a predictor instance
    predictor = Predictor(model, device)
    # Make a prediction
    predictor.predict(image)
