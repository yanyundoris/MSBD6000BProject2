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

1st Part: Converlutional layer (32 features and 3*3 filter size, SAME padding) with batch normalization and ReLU activation. No pooling layer, no drop-out layer.

2ed Part: Converlutional layer (32 features and 3*3 filter size, SAME padding) with batch normalization and ReLU activation. Have one max pooling layer with pool_size = 2, drop-out rate is 0.25.

3rd Part: Converlutional layer (64 features and 3*3 filter size, SAME padding) with batch normalization and ReLU activation. No pooling layer, drop-out rate is 0.25.

4th Part: Converlutional layer (64 features and 3*3 filter size, SAME padding) with batch normalization and ReLU activation. Have one max pooling layer with pool_size = 2, drop-out rate is 0.25.

5th Part: Flatten fully connected layer with 512 units, batch normalization layer and ReLU activation. Drop-out rate is 0.5

6th: Soft-max layer.

#### c. Some tricks I use

__Batch normalization__: Adding batch normalization layer improves my validation accuracy.

__Alternate pooling layer__: I only use max pooling in layers with odd/even index

__Drop-out__ : I set the lower drop-out rate to avoid overfitting at the first 4 parts, but in the dense fully connected layer, the drop-out rate should be a little higher since the features in this stage is more meaningful and worthy.

#### d. Validation accuracy

the final validation accuracy is 80%

### III. Stage 2: Transfer learning

Then I consider to use transfer learning method (see reference) to improve performance. I use the method provided by tensor flow in its Github, loading a pre-trained Inception v3 model, removing the old top layer, and training a new one on the given flower photos.

#### a. Pre-processing

No special preprocessing stage in my transfer learning model, just follow Inception v3 model workflow.

#### b. Train the model

1. load pre-trained Inception v3 model and flower dataset
2. calculates the bottleneck values for each image
3. train a new model on the top of bottleneck values
4. model evaluation and prediction

### d. Validation accuracy

the final validation accuracy is 90%

### IV. Model selection and evaluation

You can use my transfer learning model prediciton for grading since it achieves more higher accuracy. Meanwhile, I list the evaluation result on validation set for those two models:









