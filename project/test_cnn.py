import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from classification import CNNClassifier
from dataset_creation import generate_dataset


def test_model(model, dataloader):
    acc = 0
    for test_input, test_target in dataloader:
        image = np.transpose(test_input.squeeze().numpy(), (1, 2, 0))
        predicted = model.predict(image)
        print(f"predicted {predicted}, target {test_target.item()}")

        acc += predicted == test_target.item()
        if (predicted == 9 and test_target.item() == 6) or (predicted == 6 and test_target.item() == 9):
            acc += 1
    acc /= len(dataloader)
    return acc


def main():
    # Generate dataset
    generate_dataset(nb_samples=1000, use_only_video_dataset=False)
    test_dataset = datasets.ImageFolder(root="generated_dataset", transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1)
    print(f"Loaded {len(test_loader)} test images")

    model = CNNClassifier()

    acc = test_model(model, test_loader)
    print(f"test acc: {acc}")


if __name__ == "__main__":
    main()
