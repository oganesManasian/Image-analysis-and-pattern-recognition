from torchvision import transforms
import torch
from random import randint
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def get_stats(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False, num_workers=2)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean.item(), std.item()


def get_loaders(dataset, batch_size=100, validation_split=0.2, shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(randint(0, 100))
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader


class NormalizedDataset:

    def __init__(self, dataset):
        self.dataset = dataset
        self.mean, self.std = get_stats(self.dataset)

    def __getitem__(self, index):
        tensor, class_ = self.dataset[index]
        normalized = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        return normalized, class_

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes


class IncompleteDataset:

    def __init__(self, dataset, all_classes, mean, std):
        self.dataset = dataset
        self.all_classes = all_classes
        self.mean, self.std = mean, std

    def __getitem__(self, index):
        tensor, class_ = self.dataset[index]
        class_str = self.dataset.classes[class_]
        new_index = self.all_classes.index(class_str)
        normalized = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        return normalized, new_index

    def __len__(self):
        return len(self.dataset)
