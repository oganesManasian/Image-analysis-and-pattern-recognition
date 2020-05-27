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

GENERATED_DATASETS_DIR = "generated_dataset"
VIDEO_DATASET_DIR = "video_dataset"

WHITE = 255
BLACK = 0


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
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
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
    if validation_split > 0:
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)
    else:
        # Use same data (works since we have random rotations)
        valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler)

    return train_loader, valid_loader


def get_digit_loaders(batch_size=64, train_rotation=5, test_rotation=5):
    common_transforms = [
        transforms.Lambda(to_binary),
        transforms.ToTensor(),
        # transforms.Normalize((MEAN_DIGITS,), (STD_DIGITS,))
    ]
    common_transform = transforms.Compose(common_transforms)

    # Creating dataset with images from Mnist dataset
    minst_transform = transforms.Compose([
        transforms.RandomRotation(train_rotation, fill=(BLACK,)),
        common_transform
    ])

    mnist_dataset = datasets.MNIST("", transform=minst_transform, download=True)
    mnist_mean, mnist_std = get_stats(mnist_dataset)
    print(f"Digit's dataset mean {mnist_mean}, std {mnist_std}")
    train_loader, val_loader = get_loaders(mnist_dataset, batch_size=batch_size)

    # Creating dataset with images from video dataset
    videodataset_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.RandomRotation(test_rotation, fill=(BLACK,)),
        common_transform
    ])

    test_dataset = datasets.ImageFolder(root=f"{VIDEO_DATASET_DIR}/digits", transform=videodataset_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader


def get_operator_loaders(batch_size=1, train_rotation=5, test_rotation=5):
    common_transforms = [
        transforms.Lambda(to_binary),
        transforms.ToTensor(),
        # transforms.Normalize((MEAN_OPERATORS,), (STD_OPERATORS,))
    ]
    common_transform = transforms.Compose(common_transforms)

    # Creating dataset with images from github
    github_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        # transforms.Resize((28, 28)),
        transforms.RandomRotation(train_rotation, fill=(BLACK,)),
        common_transform
    ])

    operators_dataset = datasets.ImageFolder(root='operators', transform=github_transform)
    operators_mean, operators_std = get_stats(operators_dataset)
    print(f"Operator's dataset mean {operators_mean}, std {operators_std}")
    train_loader, val_loader = get_loaders(operators_dataset, batch_size=batch_size, validation_split=0)

    # Creating dataset with images from video dataset
    videodataset_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.RandomRotation(test_rotation, fill=(BLACK,)),
        common_transform
    ])

    test_dataset = datasets.ImageFolder(root=f"{VIDEO_DATASET_DIR}/operators", transform=videodataset_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader


def generate_dataset(data_type, nb_samples=100, use_only_video_dataset=False, train_rotation=360, test_rotation=360):
    if not os.path.isdir(GENERATED_DATASETS_DIR):
        os.mkdir(GENERATED_DATASETS_DIR)

    if data_type == "digits":
        train_loader, val_loader, test_loader = get_digit_loaders(batch_size=1,
                                                                  train_rotation=train_rotation,
                                                                  test_rotation=test_rotation)
        path = os.path.join(GENERATED_DATASETS_DIR, "digits")
        folders_to_create = map(str, range(10))
    elif data_type == "operators":
        train_loader, val_loader, test_loader = get_operator_loaders(batch_size=1,
                                                                     train_rotation=train_rotation,
                                                                     test_rotation=test_rotation)
        path = os.path.join(GENERATED_DATASETS_DIR, "operators")
        folders_to_create = ["divide", "equal", "minus", "multiply", "plus"]
        # class_ind2label = {"divide": 0, "equal": 1, "minus": 2, "multiply": 3, "plus": 4}
        class_ind2label = {key: value for key, value in enumerate(folders_to_create)}
    else:
        raise NotImplementedError

    # Delete last generated dataset
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

    for folder in folders_to_create:
        os.mkdir(os.path.join(path, folder))

    loaders = [test_loader]
    filenames = ["test"]
    if not use_only_video_dataset:
        loaders.append(train_loader)
        filenames.append("train")

    for loader, filename in zip(loaders, filenames):
        nb_generated = 0
        stop_generating = False
        while not stop_generating:
            for (img, label) in loader:
                # imsave(os.path.join(path, str(label.item()), f"train{i}.png"), img.squeeze(0).squeeze(0))
                imsave(os.path.join(path,
                                    str(label.item()) if data_type == "digits" else class_ind2label[label.item()],
                                    f"{filename}{nb_generated}.png"),
                       img.numpy().squeeze())
                nb_generated += 1
                if nb_generated >= nb_samples:
                    stop_generating = True
                    break

    print(f"{data_type} dataset created")


def preprocess_github_images(path_to_raw_data="operators github", path_to_save="operators"):
    # Prepare directory
    if os.path.isdir(path_to_save):
        shutil.rmtree(path_to_save)
    os.mkdir(path_to_save)
    folders_to_create = ["divide", "equal", "minus", "multiply", "plus"]
    for folder in folders_to_create:
        os.mkdir(os.path.join(path_to_save, folder))

    # Preprocess data
    preprocess_steps = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Pad((20, 20), fill=(WHITE, WHITE, WHITE)),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.Lambda(to_binary),
        transforms.ToTensor()
    ])
    data = datasets.ImageFolder(root=path_to_raw_data, transform=preprocess_steps)
    loader = DataLoader(data, batch_size=1)

    class_ind2label = {key: value for key, value in enumerate(folders_to_create)}

    # Save data
    for (img, label) in loader:
        # imsave(os.path.join(path, str(label.item()), f"train{i}.png"), img.squeeze(0).squeeze(0))
        imsave(os.path.join(path_to_save, class_ind2label[label.item()], f"{class_ind2label[label.item()]}.png"),
               img.numpy().squeeze())


if __name__ == "__main__":
    preprocess_github_images()
    generate_dataset(data_type="digits", nb_samples=30)
    generate_dataset(data_type="operators", nb_samples=20)
