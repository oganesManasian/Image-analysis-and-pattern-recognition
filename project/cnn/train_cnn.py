import os
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms


from cnn.networks import ConvNetSmall, ConvNet
from dataset_creation import get_digit_loaders, get_operator_loaders
# from cross_entropy import CrossEntropyLoss

from cnn.dataset_creation import generate_dataset, get_loaders, inverse_color, to_binary

METRICS_FOLDER = "metrics"
WEIGHTS_FOLDER = "weights"


def evaluate_model(model, dataloader, device):
    acc = 0
    for test_input, test_target in dataloader:
        test_input, test_target = test_input.to(device), test_target.to(device)
        predicted, _ = model.predict(test_input)

        acc_batch = (predicted == test_target).sum().item() / len(predicted)
        acc += acc_batch

    acc /= len(dataloader)
    return acc


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train(model, nb_epochs, train_loader, val_loader, test_loader, device, eval_freq=1):
    losses = []
    train_acc = []
    val_acc = []
    test_acc = []

    best_accuracy = 0
    best_model = None

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # criterion = nn.CrossEntropyLoss()
    # criterion = CrossEntropyLoss(smooth_eps=None, smooth_dist=None)
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.2)

    # Train the model
    print("Started training")
    for e in range(nb_epochs):
        epoch_loss = 0
        for train_input, train_target in train_loader:
            train_input, train_target = train_input.to(device), train_target.to(device)
            output = model(train_input)
            loss = criterion(output, train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()
        print('%dth epoch is finished and the loss is %f' % (e + 1, epoch_loss))
        losses.append(epoch_loss)

        if (e + 1) % eval_freq == 0:
            with torch.no_grad():
                model.eval()

                acc = evaluate_model(model, train_loader, device)
                train_acc.append(acc)
                print(f"train acc: {acc}")

                acc = evaluate_model(model, val_loader, device)
                val_acc.append(acc)
                print(f"val acc: {acc}")

                acc = evaluate_model(model, test_loader, device)
                test_acc.append(acc)
                print(f"test acc: {acc}")

            if test_acc[-1] >= best_accuracy:
                best_model = copy.deepcopy(model)
                best_accuracy = test_acc[-1]
                print("Best model saved")

    print(f"Best test acc reached {max(test_acc)}")
    return best_model, [losses, train_acc, val_acc, test_acc]


def plot_metrics(metrics, model_name):
    losses, train_acc, val_acc, test_acc = metrics

    plt.plot(losses)
    plt.title(f"Train loss {model_name}")
    plt.savefig(fname=f"{METRICS_FOLDER}/train loss {model_name}.png")
    plt.show()

    plt.plot(train_acc)
    plt.title(f"train acc {model_name}")
    plt.savefig(fname=f"{METRICS_FOLDER}/train acc {model_name}.png")
    plt.show()

    plt.plot(val_acc)
    plt.title(f"Val acc {model_name}")
    plt.savefig(fname=f"{METRICS_FOLDER}/val acc {model_name}.png")
    plt.show()

    plt.plot(test_acc)
    plt.title(f"Test acc {model_name}")
    plt.savefig(fname=f"{METRICS_FOLDER}/test acc {model_name}.png")
    plt.show()


def generate_model(model_name, train_loader, val_loader, test_loader, nb_classes, nb_epochs=5):
    print(f"Training {model_name} model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    if not test_loader:
        test_loader = val_loader

    print(f"Loaded {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches")

    if model_name == "digits":
        model = ConvNet(nb_classes=nb_classes).to(device)
    elif model_name == "operators":
        model = ConvNetSmall(nb_classes=nb_classes).to(device)
        # model = ConvNet(nb_classes=nb_classes).to(device)

    model, metrics = train(model, nb_epochs, train_loader, val_loader, test_loader, device)
    plot_metrics(metrics, model_name)
    torch.save(model.state_dict(), f"{WEIGHTS_FOLDER}/model_{model_name}.pth")


if __name__ == "__main__":
    if not os.path.isdir(WEIGHTS_FOLDER):
        os.mkdir(WEIGHTS_FOLDER)
    if not os.path.isdir(METRICS_FOLDER):
        os.mkdir(METRICS_FOLDER)

    # Training using dataloaders
    # train_loader, val_loader, test_loader = get_digit_loaders()
    # generate_model("digits", train_loader, val_loader, test_loader, 10, nb_epochs=5)
    # train_loader, val_loader, test_loader = get_operator_loaders()
    # generate_model("operators", get_operator_loaders, 5, nb_epochs=30)

    # Training using stored data (more convenient since we can first generate large dataset and then train on it)
    # Train digits model
    generate_dataset(data_type="digits",
                     nb_samples=4096,
                     use_only_video_dataset=False,
                     train_rotation=180,
                     test_rotation=180
                     )
    preprocess_image = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.Lambda(to_binary),
        transforms.ToTensor(),
        # transforms.Normalize((MEAN_DIGITS,), (STD_DIGITS,)),
    ])
    dataset = datasets.ImageFolder(root="generated_dataset/digits",
                                   transform=preprocess_image)
    train_loader, val_loader = get_loaders(dataset, batch_size=32)
    generate_model("digits", train_loader, val_loader, None, 10, nb_epochs=20)

    # Train operators model
    generate_dataset(data_type="operators",
                     nb_samples=4096,
                     use_only_video_dataset=False,
                     train_rotation=180,
                     test_rotation=180
                     )
    preprocess_image = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(inverse_color),
        transforms.Lambda(to_binary),
        transforms.ToTensor(),
        # transforms.Normalize((MEAN_OPERATORS,), (STD_OPERATORS,)),
    ])
    dataset = datasets.ImageFolder(root="generated_dataset/operators",
                                   transform=preprocess_image)
    train_loader, val_loader = get_loaders(dataset, batch_size=32)
    generate_model("operators", train_loader, val_loader, None, 5, nb_epochs=3)
