import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)




'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FixMatch(nn.Module):
    def __init__(self, model, T=0.5):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, X_u, y_u, X_l, y_l, X_ul, unlabeled_mask, labeled_mask):
        self.model.train()
        
        # Get predictions on unlabeled data
        logits_ul = self.model(X_ul)
        probs_ul = F.softmax(logits_ul / self.T, dim=1)
        preds_ul = torch.argmax(probs_ul, dim=1)
        
        # Create pseudo-labels for unlabeled data
        y_ul = preds_ul.detach()
        y_ul[unlabeled_mask == 0] = -1  # mask out padding values
        
        # Combine labeled and pseudo-labeled data
        X = torch.cat([X_u, X_ul], dim=0)
        y = torch.cat([y_u, y_ul], dim=0)
        
        # Train the model on combined data with labeled and pseudo-labels
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()
        logits = self.model(X)
        loss = F.cross_entropy(logits[labeled_mask], y[labeled_mask])
        loss.backward()
        optimizer.step()
        
        return loss.item()
'''