# 5013 Reproducation project: Reproduction and Fine-tuning with EfficientNetV2

The original paper: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

The open source code link: https://github.com/google/automl/tree/master/efficientnetv2

Our codes are also submitted to the canvas.

`model_training.py`: using CFAIR10 dataset to train three models.

The file fold of `fine-tuning-cars` contains `dataset.py`, `model.py`, `predict.py`, `utils.py`, `train_s.py`, `train_m.py`, `train_l.py` and relatively `trans_effv2_weights.py` of these three models.

These two datasets is about two classification datasets: Oxford 102‐category flower dataset and the Stanford Cars dataset, and using these datasets for the part of fine-tuning.

## dataset.py
This code file implements a custom PyTorch dataset class, `MyDataSet`, for loading and processing image data. The primary functionalities of this class are as follows:

#### Preloading and Validating Image Modes
During initialization, the code pre-checks the mode of all images and retains only the paths and corresponding indices of images that are in RGB mode. This ensures that only RGB images are included in the dataset.

#### Dataset Length
The `__len__` method returns the number of valid (RGB mode) images in the dataset.

#### Retrieving Data Items
The `__getitem__` method returns the image and its corresponding label based on the given index. It ensures that only images pre-verified as RGB mode are returned. If a non-RGB mode image is found during runtime (despite preloading filters), a valid RGB image is randomly selected as a replacement.

#### Data Transformation
If image transformations (such as data augmentation) are specified, they are applied to the image before returning it.

#### Batch Processing
The static method `collate_fn` is used to combine a batch of data into tensors for convenient batch processing.

#### Benefits
Through this approach, the `MyDataSet` class ensures that only RGB mode images are used during training and validation, avoiding potential mode mismatch issues and improving data loading reliability.

#### Usage
To use this custom dataset class, include the `datasets.py` file in your project and create an instance of the `MyDataSet` class with the appropriate parameters. Below is an example of how to use this class:

```python
from datasets import MyDataSet
from torchvision import transforms

# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create an instance of MyDataSet
dataset = MyDataSet(image_dir='path/to/images', labels=your_labels, transform=transform)

# Use the dataset with a DataLoader
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MyDataSet.collate_fn)

# Iterate over the DataLoader
for images, labels in data_loader:
    # Your training or validation code here
```
## model.py
This code implements the definition of the EfficientNetV2 model, including its basic building blocks and model configuration. EfficientNetV2 is an efficient convolutional neural network architecture that balances performance and efficiency through complex blocks such as MBConv and FusedMBConv. Below is a brief description of the main components and functionalities:

#### DropPath and DropPath Class:
Implements the Stochastic Depth technique, which randomly drops certain paths to enhance the model's generalization capability.
#### ConvBNAct Class: 
Defines a module containing a convolutional layer, batch normalization layer, and activation layer.
#### SqueezeExcite Class: 
Implements the Squeeze-and-Excitation mechanism, which improves model performance through adaptive feature re-weighting.
#### MBConv and FusedMBConv Classes: 
These are the basic building blocks of EfficientNetV2. MBConv uses depthwise separable convolutions and the Squeeze-and-Excitation module, while FusedMBConv combines standard convolutions with depthwise separable convolutions.
#### EfficientNetV2 Class: 
Defines the entire EfficientNetV2 model, including the main blocks and head of the model, as well as weight initialization.

## train.py
This python file is designed to train and validate the EfficientNetV2-L model using PyTorch. It includes all necessary steps for data preparation, model loading, training, and validation. Below is a detailed breakdown of the file:

#### Importing Necessary Libraries:
The file begins by importing essential libraries, including PyTorch, libraries for data augmentation and processing, model and dataset definitions, utility functions, and libraries for logging and visualizing the training process.

#### Main Function:
The main function is the core of the training process, encompassing the key steps for model training and validation.

#### Argument Parsing:
Uses the argparse library to parse command-line arguments, including:

> num_classes: Number of classes for classification.

> epochs: Number of training epochs.

> batch-size: Batch size.

> lr: Initial learning rate.

> lrf: Learning rate decay factor.

> data-path: Root directory of the dataset.

> weights: Path to the pre-trained model weights.

> freeze-layers: Whether to freeze certain layers of the model.

> device: Specifies the device to use (e.g., GPU or CPU).

#### Data Preparation:
Uses the read_split_data function to read and split the training and validation datasets.
Defines data augmentation and preprocessing methods.
Instantiates the training and validation datasets, along with the corresponding data loaders.

#### Model Configuration:
Creates an instance of the EfficientNetV2-L model and loads it onto the specified device.
Loads pre-trained weights if a path is specified.
Optionally freezes certain layers of the model based on the parameters.

#### Optimizer and Learning Rate Scheduler:
Uses the SGD optimizer, configuring the learning rate, momentum, and weight decay.
Uses a cosine annealing learning rate scheduler to adjust the learning rate based on the training progress.

#### Training and Validation Loop:
Initializes lists to store validation losses and accuracies.
Iterates over each training epoch to perform training and validation:
Uses the train_one_epoch function for a single epoch of training.
Updates the learning rate using the scheduler.

## trans_effv2_weights.py
The primary function of this code is to convert TensorFlow model weights to PyTorch model weights, enabling the use of a pre-trained TensorFlow model in a PyTorch environment. Specifically, the code reads weights from a TensorFlow checkpoint file, converts the shapes and names of the weights as necessary, and then saves the converted weights as a PyTorch .pth file.

# The result of fine-tuning
<img width="489" alt="image" src="https://github.com/carachu1/5013/assets/150044043/aa8c549a-697b-40b7-b6cd-ffc151a39c1f">

