# TDT4265-Computer-Vision-and-Deep-Learning

Assignments and project from the course TDT4265 Computer vision and Deep Learning @ NTNU. 

## Assignments

#### Assignment 1

Simple single layer neural network used classifying handwritten digits on the MNIST dataset.

#### Assignment 2

More complex multilayer neural network for classifying handwritten digits on the MNIST dataset.

#### Assignment 3

Convolutional neural networks (CNNs) and pretrained CNNs used for classification of various objects in images.

#### Assignment 4

Object detection using a single shot multibox detector (SSD).

## TDT4265-Project
Detecting wear and tear on roads using a single shot detector. Project related to the course TDT4265 - Computer Vision and Deep Learning @ NTNU.

### Abstract
This Project folder contains code for a single shot multibox detector trained to detect different types of road damages on Norwegian roads. The folder contains two SSD models

- A model that has been pretrained on the [RDD2020](https://paperswithcode.com/dataset/rdd-2020) dataset, a dataset containing 26000 labeled images from around the world. The pretraining has been done for 16k iterations lasting 8 hours on a NVIDIA T4 card. The pretrained model has been further trained on a private dataset containing labeled images from Norwegian roads. This model has been trained for another 18k iterations, lasting 11 hours on a NVIDIA T4 card. The final model reached a mAP of 0.03, comfortably beating the baseline target model (0.028).

- A Feature-Fused SSD model, which has been developed to further increase performance on the RDD2020 dataset. The model reaced a mAP of 0.44 after 20k iterations, significantly exceeding the baseline target model (0.36 mAP)

### Running the code
Install the required dependencies

```
pip install -r requirements.txt
```

Navigate to the SSD folder
```
cd BaseModel/SSD
```

To train the model, simply execute
```
python train.py configs/<config_name>
```
where ```<config_name>``` is replaced by one of the following:

- ```train_tdt4265.yaml``` for training the first model on the Norwegian dataset (given you have access).
- ```train_rdd2020_server.yaml``` for training the first model on the RDD2020 dataset.
- ```train_rdd2020_fused_ssd_server.yaml``` for training the Fused SSD on the RDD2020 dataset.
