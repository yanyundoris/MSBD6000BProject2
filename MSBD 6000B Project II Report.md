# MSBD 6000B Project II Report

##### Student Name: LIU Yan Yun       Student Id: 20384933

## I. Introduction

　In this project, I try two different method to classify flower photoes. Firstly, I train and evaluate a CNN model with training and validation sets, the validation accuracy is about 79%, which is not good enough. Training an entire CNN model from scratch cannot achive a satisfactory performance since we only have limited data (2500+ for training and 500+ for validation). So I consider to use a pretrained model with a large scale of data, and treat the last layer output as initialization for my prediction model. By following this transfer learning workflow, my validation accuracy increased to 90%.

### II. Stage 1: Training a CNN model with random initialization

#### a. Pre-processing

__Resize image:__ The row input image is too large and not unique. I need to resize those images with a fixed size, and the size of image should not be too large so my laptop can handle. I resize the image into 

__Normalization:__ Before training , I normalize all image arrays into range 0-1

__One hot encoding__: For labels, I use one hot encoding to transform them into vectors.

#### b. Train CNN model

The model strucuture is shown below:



#### c. Some tricks I use

__Batch normalization__: Adding batch normalization layer improves my validation accuracy.

__Alternate pooling layer__: I only use max pooling in layers with odd/even index

#### d. Save the model

After training, I save the 

the final validation accuracy is 80%

### IV. Stage 2: Transfer learning

Then I consider to use transfer learning method (see reference) to improve performance. I use the method provided by tensor flow in its Github, loading a pre-trained Inception v3 model, removing the old top layer, and training a new one on the given flower photos.

1. load pre-trained Inception v3 model and flower dataset
2. calculates the bottleneck values for each image
3. train a new model on the top of bottleneck values
4. model evaluation and prediction

the final validation accuracy is 90%





