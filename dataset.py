import os

import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset


class HiRiseDataset(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotations.iloc[index, 0])
        image = io.imread(image_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if (self.transform):
            image = self.transform(image)

        return (image, y_label)
