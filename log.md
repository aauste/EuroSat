#### 2026-09-16
---
Tried using ImageFolder to load the dataset. Had difficulty figuring out how to use the existing train/val/test split and label map provided by Kaggle whilst loading, so decided to swap to creating my own Dataset class, which takes a csv path to the necessary splits and creates the individual datasets.
Started exploring the RGB dataset by looking into the size of the splits, how the tensors of images look like and plotting samples of the different classes.