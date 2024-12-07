import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        data_arr = self.data.to_numpy()
        labels = data_arr[:, 0]
        images = data_arr[:, 1:].reshape(-1, 28, 28, 1) / 255  # Normalize and reshape
        return train_test_split(images, labels, test_size=0.2, random_state=42)
