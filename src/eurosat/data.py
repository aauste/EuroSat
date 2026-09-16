import kagglehub
import pandas as pd
import os
import json
from torchvision.transforms import v2
from torchvision.io import decode_image

def download_data():
    path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
    print(f'Data at {path}.')

    return path

def load_data_rgb(path):
    train_data = EuroSATRGB(f'{path}/EuroSAT/train.csv', path)
    val_data = EuroSATRGB(f'{path}/EuroSAT/validation.csv', path)
    test_data = EuroSATRGB(f'{path}/EuroSAT/test.csv', path)

    return train_data, val_data, test_data
    
class EuroSATRGB():
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path, index_col=0).sort_index()
        self.root = f'{root_dir}/EuroSAT'
        self.transform = transform
        with open(f'{self.root}/label_map.json') as f:
            file = json.load(f)
            self.label_map = {v: k for k, v in file.items()}
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx]['Filename'])
        image = decode_image(img_path) # Returns a tensor.
        image = v2.functional.to_dtype(image, scale=True) # Scales data 0-1, done for all split sets.
        label = self.df.iloc[idx]['Label']

        if self.transform: # Additional specified transforms, normalization and ToTensor is assumed.
            image = self.transform(image)

        return image, label