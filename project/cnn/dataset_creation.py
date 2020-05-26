import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import PIL
import os
import shutil
from torch.utils.data.sampler import SubsetRandomSampler
from random import randint
from skimage.io import imsave

MEAN_DIGITS = 0.8694
STD_DIGITS = 0.3303

PATH = "generated_dataset"


def inverse_color(img):
    return PIL.Image.eval(img, lambda val: 255 - val)


def fix_background_color_bug(img):
    colors = sorted(img.getcolors(), key=lambda pair: pair[0], reverse=True)
    replace_color = colors[0][1]
    remove_color = colors[2][1] if colors[2][1] > colors[1][1] else colors[1][1]

    data = np.array(img)
    data[data == remove_color] = replace_color
    return PIL.Image.fromarray(data)


def to_binary(img):
    return PIL.Image.eval(img, lambda val: 255 if val < (256 / 2) else 0)


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

    return mean, std


def get_loaders(dataset, batch_size, validation_split=0.2, shuffle_dataset=True):
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


def get_digit_loaders(batch_size=64, train_rotation=10, test_rotation=360):
    white = 255
    black = 0

    common_transforms = [
        transforms.Lambda(to_binary),
        transforms.ToTensor(),
        # transforms.Normalize((MEAN_DIGITS,), (STD_DIGITS,))
    ]

    common_transform = transforms.Compose(common_transforms)

    minst_tensor_transform = transforms.Compose([
        transforms.RandomRotation(train_rotation, fill=(black,)),
        common_transform
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.RandomRotation(test_rotation, fill=(black,)),
        common_transform
    ])

    # Minst Dataset
    minst_dataset = datasets.MNIST("", transform=minst_tensor_transform, download=True)
    minst_mean, minst_std = get_stats(minst_dataset)
    print(f"Digit's dataset mean {minst_mean}, std {minst_std}")
    train_loader, val_loader = get_loaders(minst_dataset, batch_size=batch_size)

    test_dataset = datasets.ImageFolder(root="video_dataset/digits", transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader


def get_operators_loaders():
    ## Adding Grayscale + Inverse color to operators
    # operators_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Lambda(inverse_color),
    #     # Randomly scale up and down
    #     #     transforms.RandomAffine(0, scale=(0.9, 1.1), fillcolor=white),
    #     common_transform,
    # ])

    # operators_tensor_transform = transforms.Compose([
    #     operators_transform,
    #     transforms.ToTensor(),
    # ])

    # # Operators Dataset
    # operators_dataset = datasets.ImageFolder(root='operators', transform=operators_tensor_transform)
    # operators_mean, operators_std = get_stats(operators_dataset)
    # print(operators_mean, operators_std)
    pass


def generate_dataset(nb_samples=1000, use_only_video_dataset=True, train_rotation=360, test_rotation=360):
    train_loader, val_loader, test_loader = get_digit_loaders(batch_size=1,
                                                              train_rotation=train_rotation,
                                                              test_rotation=test_rotation)

    if os.path.isdir(PATH):
        shutil.rmtree(PATH)

    os.mkdir(PATH)
    for i in range(10):
        os.mkdir(os.path.join(PATH, str(i)))

    if use_only_video_dataset:
        for i, (img, label) in enumerate(train_loader):
            # imsave(os.path.join(PATH, str(label.item()), f"train{i}.png"), img.squeeze(0).squeeze(0))
            imsave(os.path.join(PATH, str(label.item()), f"train{i}.png"), img.numpy().squeeze())
            if i > nb_samples:
                break

    nb_generated = 0
    while nb_generated < nb_samples:
        for i, (img, label) in enumerate(test_loader):
            # imsave(os.path.join(PATH, str(label.item()), f"test{nb_generated + i}.png"), img.squeeze(0).squeeze(0))
            imsave(os.path.join(PATH, str(label.item()), f"test{nb_generated + i}.png"), img.numpy().squeeze())
        nb_generated += i
