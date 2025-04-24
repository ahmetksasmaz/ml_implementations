import numpy as np
import cv2 as cv

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
    
    def load_data(self):
        """
        Load the MNIST dataset from the specified path.
        The dataset should be in a format that can be read by the parser.
        """
        with open(self.data_path, "r") as f:
            # Read the file and skip the header
            rows = f.readlines()[1:]

            for row in rows:
                # Split the row into label and pixel values
                columns = row.split(",")
                label = int(columns[0])
                pixels = [int(x) for x in columns[1:]]
                
                # Convert the pixel values to a numpy array and reshape it to 28x28
                image = np.array(pixels, dtype=np.uint8).reshape(28, 28)
                image = image.astype(np.float32) / 255.0
                # Append the image and label to the data list
                self.data.append((image, label))
    
    def shuffle_data(self):
        """
        Shuffle the data to ensure randomness.
        """
        np.random.shuffle(self.data)
    
    def get_data(self):
        """
        Return the loaded data.
        """
        return self.data
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Return the image and label at the specified index.
        """
        return self.data[index]
    
    def get_batches(self, batch_size):
        """
        Return batches of data of the specified size.
        """
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i + batch_size]