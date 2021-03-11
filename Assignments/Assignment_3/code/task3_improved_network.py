import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from trainer import Trainer, compute_loss_and_accuracy
from task2 import ExampleModel, create_plots

from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import typing
import torch
import numpy as np


mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def load_cifar10_augmented(batch_size: int, validation_fraction: float = 0.1) -> typing.List[DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
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


class ConvModel(ExampleModel):
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__(image_channels, num_classes)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        kernel = 5
        activation_func = nn.ReLU
        dropout_p = 0.05

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            activation_func(),

            # 2
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d([2, 2], stride=2),
            activation_func(),

            # 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            activation_func(),

            # 4
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            nn.MaxPool2d([2, 2], stride=2),
            nn.Dropout2d(p=dropout_p),
        )

        # Initialize our last fully connected layer
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 128*8*8
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.BatchNorm1d(64),
            activation_func(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),

            nn.Linear(64, num_classes),
        )


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 3.5e-2
    early_stop_count = 4
    dataloaders = load_cifar10_augmented(batch_size)
    # Use SGD
    model = ConvModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    trainer.test_model()

    create_plots(trainer, "task3e")
