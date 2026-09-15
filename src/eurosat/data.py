import kagglehub
import pandas as pd
import json
from torchvision.datasets import ImageFolder
from torchvision import transforms

def download_data():
    path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
    print(f'Data at {path}.')

    return path

class RGBData():
    def __init__(self, path):
        self.path = f'{path}/EuroSAT'
        self.dataset = None
        self.label_map = None

    def load_data(self):
        self.dataset = ImageFolder(root=self.path, transform=transforms.ToTensor())
        with open(f'{self.path}/label_map.json') as f:
            self.label_map = json.load(f)

        return self.dataset, self.label_map