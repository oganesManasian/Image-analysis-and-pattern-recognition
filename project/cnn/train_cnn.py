import os
import shutil

import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

from classification import Conv_Net
from dataset_creation import get_digit_loaders
# from cross_entropy import CrossEntropyLoss

CNN_PATH = "cnn"


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
    val_acc = []
    test_acc = []

    # best_accuracy = 0
    # best_model = None

    lr = 5e-3
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
            acc = evaluate_model(model, val_loader, device)
            val_acc.append(acc)
            print(f"val acc: {acc}")

            n_rounds = 100
            acc = 0
            for _ in range(n_rounds):
                acc += evaluate_model(model, test_loader, device)
            acc /= n_rounds
            test_acc.append(acc)
            print(f"test acc: {acc}")

            # if test_acc[-1] > best_accuracy:
            #     best_model = copy.deepcopy(model)
            #     best_accuracy = test_acc[-1]

    plt.plot(losses)
    plt.title("Train loss")
    plt.savefig(fname=f"{CNN_PATH}/train_loss.png")
    plt.show()

    plt.plot(val_acc)
    plt.title("Val acc")
    plt.savefig(fname=f"{CNN_PATH}/val_acc.png")
    plt.show()

    plt.plot(test_acc)
    plt.title("Test acc")
    plt.savefig(fname=f"{CNN_PATH}/test_acc.png")
    plt.show()

    return model


def main():
    if os.path.isdir(CNN_PATH):
        shutil.rmtree(CNN_PATH)
    os.mkdir(CNN_PATH)

    nb_epochs = 10
    train_loader, val_loader, test_loader = get_digit_loaders()
    print(f"Loaded {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)
    model = Conv_Net(nb_classes=10).to(device)

    model = train(model, nb_epochs, train_loader, val_loader, test_loader, device)

    torch.save(model.state_dict(), f"{CNN_PATH}/cnn_model.pth")


if __name__ == "__main__":
    main()
