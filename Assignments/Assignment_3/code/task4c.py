
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
from task4b import torch_image_to_numpy


def plot_conv_output(model, model_in, model_out):
    f, axarr = plt.subplots(1, 10)
    for i in range(10):
        axarr[i].imshow(torch_image_to_numpy(model_out[0][i]), cmap="gray")
    plt.show()


if __name__ == "__main__":

    image = Image.open("assignment3/images/zebra.jpg")
    print("Image shape:", image.size)

    # Resize, and normalize the image with the mean and standard deviation
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)[None]
    print("Image shape:", image.shape)

    # Load the model
    model = torchvision.models.resnet18(pretrained=True)
    print(model)
    # Extract the last conv laer
    last_conv_layer = model.layer4[1].bn2
    # Add hook
    last_conv_layer.register_forward_hook(plot_conv_output)
    # Pass zebra through network
    model(image)
