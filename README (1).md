# 🌸 Flower Classification using ResNet-50 (PyTorch)

## Overview

This project implements an image classification model for identifying
**102 flower species** using transfer learning with a pretrained
**ResNet-50** network.

The model is trained on the **Oxford 102 Flowers dataset**, which
contains **8,189 images across 102 flower categories**.

The goal of the project is to demonstrate how transfer learning can
efficiently train deep learning models for image classification.

------------------------------------------------------------------------

## Dataset

Dataset: **Oxford 102 Flower Dataset**

Total images: **8,189**\
Number of classes: **102 flower species**

Official dataset split:

  Dataset      Images
  ------------ --------
  Train        1020
  Validation   1020
  Test         6149

Dataset files:

-   `102flowers/` -- flower images
-   `imagelabels.mat` -- image labels
-   `setid.mat` -- train/validation/test split

------------------------------------------------------------------------

## Model Architecture

The project uses **ResNet-50 pretrained on ImageNet**.

Transfer learning pipeline:

1.  Load pretrained ResNet-50
2.  Freeze feature extraction layers
3.  Replace final fully connected layer
4.  Train classifier for **102 flower classes**

Architecture:

ResNet50 Backbone (Frozen)\
↓\
Fully Connected Layer\
↓\
102 Flower Classes

------------------------------------------------------------------------

## Technologies Used

-   Python
-   PyTorch
-   Torchvision
-   NumPy
-   SciPy
-   Matplotlib
-   Google Colab

------------------------------------------------------------------------

## Data Preprocessing

Images are transformed using:

-   Resize to **224×224**
-   Random horizontal flip (augmentation)
-   Tensor conversion
-   Normalization using ImageNet statistics

------------------------------------------------------------------------

## Training Setup

  Parameter       Value
  --------------- ------------------
  Model           ResNet-50
  Optimizer       Adam
  Loss Function   CrossEntropyLoss
  Batch Size      32
  Epochs          5
  Input Size      224×224

------------------------------------------------------------------------

## Results

  Metric                Score
  --------------------- ------------
  Validation Accuracy   **78.63%**
  Test Accuracy         **74.74%**

The model achieves reasonable performance with minimal training.
Accuracy can be improved with longer training and additional
fine‑tuning.

------------------------------------------------------------------------

## Project Structure

    flower-classification-resnet50/

    │
    ├── flower_classification.ipynb
    ├── README.md
    ├── imagelabels.mat
    ├── setid.mat
    └── 102flowers/

------------------------------------------------------------------------

## How to Run

### Clone repository

    git clone https://github.com/yourusername/flower-classification-resnet50.git
    cd flower-classification-resnet50

### Install dependencies

    pip install torch torchvision matplotlib scipy tqdm

### Run notebook

    jupyter notebook flower_classification.ipynb

------------------------------------------------------------------------

## Future Improvements

Potential improvements:

-   Train for **15--20 epochs**
-   Unfreeze deeper ResNet layers for fine‑tuning
-   Add stronger data augmentation
-   Use learning rate scheduling
-   Experiment with other architectures (EfficientNet, ViT)

