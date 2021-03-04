from torchvision import transforms, datasets, models
import torchvision
import torch
import utils
import typing
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from trainer import Trainer
import numpy as np


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_cifar10(batch_size: int, validation_fraction: float = 0.1) -> typing.List[DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = DataLoader(data_train,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  drop_last=True)

    dataloader_val = DataLoader(data_train,
                                sampler=validation_sampler,
                                batch_size=batch_size,
                                num_workers=2)

    dataloader_test = DataLoader(data_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test


class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters():  # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():  # Unfreeze the last fully-connected
            param.requires_grad = True  # layer
        for param in self.model.layer4.parameters():  # Unfreeze the last 5 convolutional
            param.requires_grad = True  # layers

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    epochs = 5
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = TransferModel()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task4a")
