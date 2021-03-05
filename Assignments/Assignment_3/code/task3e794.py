%matplotlib inline
# -*- coding: utf-8 -*-
# Things to try:

"""
• Data Augmentation: Data augmentation is a simple trick to extend your training set. To use
this, get familiar with the torchvision transforms abstraction.

• Filter size: The starting architecture has 5x5 filters; would small filter sizes work better?

• Number of filters: The starting architecture has 32, 64 and 128 filters. Do more of less work
better?

• Pooling vs strided convolutions: Pooling is used to reduce the input shape in the width and
height dimension. Strided convolution can also be used for this (S > 1).

• Batch normalization: Try adding spatial batch normalization after convolution layers and 1-
dimensional batch normalization after fully-connected layers. Do your networks train faster?
    -CHECK

• Network architecture: The network above has two layers of trainable parameters. Can you do
better with a deep network? Good architectures to try include:
– (conv-relu-pool)xN → (affine)xM → softmax
– (conv-relu-conv-relu-pool)xN → (affine)xM → softmax
– (batchnorm-relu-conv)xN → (affine)xM → softmax

• Regularization: Add L2 weight regularization, or perhaps use Dropout.

• Optimizers: Try out a different optimizer than SGD.

• Activation Functions: Try replacing all the ReLU activation function.
"""


import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from trainer import Trainer, compute_loss_and_accuracy
from task2 import ExampleModel, create_plots
import dataloaders as dl

from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import typing
import numpy as np


mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


# Used to plot two models in task 3d.
def create_plots2(trainer1: Trainer,trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    #plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss with and without data augmentation.")
    plt.xlabel("Training steps")
    plt.ylabel("Cross Entropy Loss")
    
    # Plot 1 model
    utils.plot_loss(
        trainer1.train_history["loss"], label="No data aug: Training loss", npoints_to_average=10)
    utils.plot_loss(
        trainer1.validation_history["loss"], label="No data aug: Validation loss")

    utils.plot_loss(
        trainer2.train_history["loss"], label="Data aug: Training loss", npoints_to_average=10)
    utils.plot_loss(
        trainer2.validation_history["loss"], label="Data aug: Validation loss")
    plt.legend()

    """ Dont need two plots
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(
        trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    """
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

# Loads augmented CIFAR dataset.
def load_cifar10_augemted(batch_size: int, validation_fraction: float = 0.1) -> typing.List[DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
    transform_train = transforms.Compose([
        # #       Randomly apply augmentations
        #         transforms.RandomApply([
        #             transforms.RandomCrop(32, padding=4),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.RandomRotation(10),
        #             transforms.RandomPerspective(),
        #             transforms.ColorJitter(0.5,0.5,0.5,0.5),
        #         ], p=0.5),
        # transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
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
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            activation_func(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            nn.MaxPool2d([2, 2], stride=2),


            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            activation_func(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            nn.Dropout2d(p=dropout_p),
            nn.MaxPool2d([2, 2], stride=2),
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
            nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            nn.Dropout(p=dropout_p),

            nn.Linear(64, num_classes),
        )

class ConvModelThickLayer(ExampleModel):
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
        kernel = 3
        activation_func = nn.ReLU
        dropout_p = 0
        
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            activation_func(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=32,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            nn.MaxPool2d([2, 2], stride=2),
            
            nn.BatchNorm2d(32),
            activation_func(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            nn.MaxPool2d([2, 2], stride=2),

            
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            activation_func(),
            
            
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            activation_func(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=kernel,
                stride=1,
                padding=2
            ),
            activation_func(),
            #nn.Dropout2d(p=dropout_p),
            nn.MaxPool2d([2, 2], stride=2),
        )

        # Initialize our last fully connected layer
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = (128*8*8)
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            #nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            #nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            #nn.Dropout(p=dropout_p),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            activation_func(),
            #nn.Dropout(p=dropout_p),

            nn.Linear(64, num_classes),
        )

        
if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 4e-2
    early_stop_count = 10

    # We train two models, with and witihout data augmentation

    # Model 1
    print("Training model 1: with thickkkkk")
    dataloaders = load_cifar10_augemted(batch_size)
    model = ConvModelThickLayer(image_channels=3, num_classes=10)
    trainer_no_aug = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer_no_aug.train()
    trainer_no_aug.test_model()
    """
    # Model 2
    print("Training model 2: with dropout")
    dataloaders = load_cifar10_augemted(batch_size)
    model = ConvModel(image_channels=3, num_classes=10)
    trainer_aug = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer_aug.train()
    trainer_aug.test_model()
    """
    
    
    
    #create_plots2(trainer_no_aug,trainer_aug, "task3d")
