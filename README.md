# VAE
A Variational Autoencoder based on FC (fully connected) and FCN (fully convolutional) architecture, implemented in PyTorch.

# Requirements
python3.8

torch==1.6.0

torchvision==1.7.0

numpy==1.18.5


# Dataset
MNIST dataset automatically downloaded from torchvision.datasets.MNIST


# Experiments
To run the FC model, execute the following:
python main.py --use_FC=1 

To run the FCN model, execute the following:
python main.py --use_FC=0

