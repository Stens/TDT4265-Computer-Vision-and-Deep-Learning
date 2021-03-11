
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
from task4b import torch_image_to_numpy


def plot_conv_output(model, model_in, model_out):
    zebra = Image.open("assignment3/images/zebra.jpg")
    f, axarr = plt.subplots(2, 5)
    for i in range(5):
        activation = (torch_image_to_numpy(
            model_out[0][i])*255).astype(np.uint8)
        activation = Image.fromarray(activation)
        activation = activation.resize((224, 224), resample=Image.NEAREST)
        axarr[0, i].imshow(activation)
        axarr[0, i].imshow(zebra, alpha=0.5)
        axarr[0, i].axis('off')
    j = 5
    for i in range(5):
        activation = (torch_image_to_numpy(
            model_out[0][j])*255).astype(np.uint8)
        activation = Image.fromarray(activation)
        activation = activation.resize((224, 224), resample=Image.NEAREST)
        axarr[1, i].imshow(activation)
        axarr[1, i].imshow(zebra, alpha=0.5)
        axarr[1, i].axis('off')
        j += 1

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
