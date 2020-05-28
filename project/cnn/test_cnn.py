import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from classification import CNNClassifier
from dataset_creation import generate_dataset


GENERATED_DATASETS_DIR = "generated_dataset"


def get_accuracy(model, dataloader):
    acc = 0
    for test_input, test_target in tqdm(dataloader):
        image = np.transpose(test_input.squeeze().numpy(), (1, 2, 0))
        predicted = model.predict(image, return_raw_label=True)
        # print(f"predicted {predicted}, target {test_target.item()}")

        acc += predicted == test_target.item()

        # Do thing which will be done later by postprocess_predicted_sequence method
        if (predicted == 9 and test_target.item() == 6) or (predicted == 6 and test_target.item() == 9):
            acc += 1
    acc /= len(dataloader)
    return acc


def test_model(data_type, nb_samples=1000, use_only_video_dataset=True):
    print(f"Testing {data_type} model")
    # Generate dataset
    generate_dataset(data_type=data_type, nb_samples=nb_samples, use_only_video_dataset=use_only_video_dataset)
    # preprocess_image = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Lambda(inverse_color),
    #     transforms.Lambda(to_binary),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((,), (,)),
    # ])
    test_dataset = datasets.ImageFolder(root=f"{GENERATED_DATASETS_DIR}/{data_type}",
                                        transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1)
    print(f"Loaded {len(test_loader)} test images")

    model = CNNClassifier(data_type=data_type)

    acc = get_accuracy(model, test_loader)

    return acc


if __name__ == "__main__":
    acc_digits, acc_operators = None, None

    # acc_digits = test_model(data_type="digits", nb_samples=512, use_only_video_dataset=False)
    acc_operators = test_model(data_type="operators", nb_samples=512, use_only_video_dataset=False)

    print(f"Test accuracy digits: {acc_digits}, operators: {acc_operators}")
