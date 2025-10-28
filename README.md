# mee-ept-sdm

Code for the paper: "How to use unmanned aerial vehicle imagery and artificial intelligence to model species distribution in a spatially restricted area".  
It enables the data creation, data processing, model training and model inference to develope species distribution modeling (SDM) for aquatic emerging insects (EPT: Ephemeroptera, Plecoptera, Trichoptera).

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Main Features](#main-features)
- [Model description](#model-description)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview

This repository provides a pipeline for:
- Creating variables and extracting features from geospatial (digital elevation models, multispectral data and vectors)
- Generating patches from rasters or raw images
- Training and evaluating machine learning models (Random Forest, MLP, ResNet, Custom CNN) for SDM tasks
- Creating prediction maps

## Repository Structure

```
├── modelling/
│   ├── Specific_CNN_plecoptera_example.ipynb
│   ├── RF_plecoptera_example.ipynb
│   ├── config.py
│   ├── emissions.csv
│   ├── image_preprocessing.py
│   ├── inference.py
│   ├── labels.csv
│   ├── models.py
│   ├── utils.py
├── patch_creation/
│   ├── patch_creation.py
│   ├── remote_sensing_utils.py
├── variable_creation/
│   ├── derived_raster_extraction.py
│   ├── derived_tabular_variables_extraction.py
│   ├── utils.py
├── requirements.txt
├── environment.yml
```

### Key Directories

- **modelling/**: Notebooks and scripts for model training, inference, configuration, and preprocessing.
- **patch_creation/**: Scripts for creating spatial data patches from rasters or raw images.
- **variable_creation/**: Scripts for creating tabular variables or raster from digital elevation models, multispectral data and vectors.

## Main Features

- **End-to-end SDM pipeline**: From raw data to predictions, evaluation and inference.
- **Custom and standard models**: Includes Random Forest, MLP, ResNet and Specific CNN approaches.
- **Geospatial processing**: Patch creation and raster variable extraction utilities.
- **Reproducible environments**: Provided via `environment.yml` and `requirements.txt`.

## Model description

#### Random Forest

The Random Forest algorithm (Breiman 2001) is a learning method that builds a large number of decision trees, each trained on a random subset of the data and variables. The individual trees then vote for a class, and the majority vote determines the final prediction. This approach is known to reduce overfitting and increase robustness compared to a single decision tree. Random Forests naturally handle tabular input data. Therefore, for each EPTs order, we provide the derived tabular dataset as input to the Random Forest. To improve model efficiency, we first use the SelectFromModel module from Scikit-learn (Pedregosa et al., 2011) to remove the least important variables, and then train the model on the remaining ones. Following R. Gerber et al. (2023), we set the number of trees to 10,000.

#### Multilayer perceptron

The multilayer perceptron (MLP) is a type of feedforward neural network commonly used in deep learning for non-linear statistical modeling. An MLP consists of an input layer, several hidden layers, and an output layer that produces the final prediction. Each hidden layer applies a non-linear activation function, here the ReLU (rectified linear unit), which allows the network to capture complex relationships between input variables. To prevent overfitting, each layer is equipped with an L2 kernel regularizer, which penalizes large weights and encourages simpler models. Our MLP architecture has one input layer with 256 neurons, followed by three hidden layers with 128, 64, and 32 neurons, respectively. The output layer has a single neuron with a sigmoid activation function to produce predictions between 0 and 1. This architecture is particularly efficient with tabular data. As a consequence, the derived tabular dataset is provided as input to the network.

#### ResNet-50

ResNet-50 (He et al. 2016) is a deep convolutional neural network (CNN) designed to specifically process image data, including rasters. It is composed of 50 layers, designed to overcome the vanishing and exploding gradient problems that often hinder the training of very deep networks. These problems occur because gradients used to update the network weights can become extremely small or large, preventing effective learning. ResNet introduces residual blocks, which include skip connections that allow the input of a block to bypass one or more layers and be added directly to its output. This enables the network to learn residual functions instead of full transformations, simplifying the optimization and allowing much deeper architectures to be trained.

Convolutional layers in ResNet-50 apply small filters across the input image to capture local spatial patterns, while deeper layers combine these local features into more abstract, global representations. Non-linear activation functions, such as ReLU, are applied after each convolution to model complex relationships between pixels. By stacking multiple residual blocks, ResNet-50 can automatically learn hierarchical feature representations, from simple edges and textures to complex spatial structures, which are particularly useful for satellite or UAV-derived images. Therefore, the ResNet-50 architecture has been increasingly used to train species distribution models from satellite-derived rasters. In our study, we first feed the ResNet-50 with the handcrafted derived raster dataset, and with the unmanipulated image patches from the raw dataset.

#### Specific CNN

The Specific CNN has a multi-branch architecture which efficiently captures spatial features at different scales. The network consists of three branches, each containing two convolutional blocks. Each block is composed of a convolutional layer followed by a max-pooling layer. Convolutional layers apply learnable filters to extract local patterns from the input, while max-pooling layers reduce the spatial resolution, retaining the most prominent features and providing some translational invariance. Batch normalization layers are inserted between the two blocks to normalize the activations, which stabilizes and accelerates training by reducing internal covariate shift. At the end of each branch, a 2D global average pooling layer aggregates the feature maps into a single vector per channel, summarizing the extracted information. The three branches differ in the kernel size of their convolutional layers—3, 5, and 7 pixels—allowing the network to detect patterns of varying sizes.

The outputs of the three branches are concatenated and passed to the classification head. This head includes a dense layer with 256 neurons and a ReLU activation, followed by a dropout layer with a probability of 0.5. Dropout randomly sets a fraction of the neurons to zero during training, which prevents overfitting by encouraging the network to develop redundant representations. Another dense layer with 128 neurons and ReLU activation, followed by a dropout layer with probability 0.3, further processes the features before the final single-neuron output layer with a sigmoid activation produces the prediction. Compared to ResNet-50, this specific network is much lighter (~270,000 parameters vs ~23,000,000), making it more suitable for datasets with a limited number of samples. As with ResNet-50, the network is trained on the derived raster dataset and on the raw dataset.

---
- Breiman, Leo. 2001. « Random Forests ». Machine Learning 45 (1): 5‑32. https://doi.org/10.1023/A:1010933404324.
- Gerber, Rémi, Christophe Piscart, Jean-Marc Roussel, et al. 2023. « Landscape Models Can Predict the Distribution of Aquatic Insects across Agricultural Areas ». Landscape Ecology 38 (11): 2917‑29. https://doi.org/10.1007/s10980-023-01761-4.
- He, Kaiming, Xiangyu Zhang, Shaoqing Ren, et Jian Sun. 2016. « Deep Residual Learning for Image Recognition ». 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), juin, 770‑78. https://doi.org/10.1109/CVPR.2016.90. 
- Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, et al. 2011. « Scikit-learn: Machine Learning in Python ». J. Mach. Learn. Res. 12 (null): 2825‑30. 

## Getting Started

Clone the repository:
```bash
git clone https://github.com/augustin-delabrosse/mee-ept-sdm.git
cd mee-ept-sdm
```

Set up the environment (recommended):
```bash
conda env create -f environment.yml
conda activate test-env
```
Or install requirements using pip:
```bash
pip install -r requirements.txt --no-deps
```

## Usage

- Explore notebooks under `modelling/` for training and evaluating models.
- Use scripts in `patch_creation/` and `variable_creation/` for data preparation and feature extraction.

## Requirements

See [requirements.txt](https://github.com/augustin-delabrosse/mee-ept-sdm/blob/main/requirements.txt) or [environment.yml](https://github.com/augustin-delabrosse/mee-ept-sdm/blob/main/environment.yml) for the full list of dependencies.

---
For more details, see the individual script and notebook documentation.