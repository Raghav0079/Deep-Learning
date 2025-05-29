# Machine Learning and Data Science Experiments in Colab

This repository contains a Google Colab notebook showcasing various machine learning and data science experiments. The notebook covers topics ranging from object detection using YOLOv4 to text generation with Recurrent Neural Networks.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Setup and Running the Notebook](#setup-and-running-the-notebook)
    *   [Prerequisites](#prerequisites)
3.  [Code Sections](#code-sections)
    *   [YOLOv4 Object Detection](#yolov4-object-detection)
    *   [Iris Dataset Classification](#iris-dataset-classification)
    *   [MNIST Handwritten Digit Recognition](#mnist-handwritten-digit-recognition)
    *   [Fashion MNIST Image Classification](#fashion-mnist-image-classification)
    *   [Text Generation with RNNs](#text-generation-with-rnns)
    *   [COCO Dataset Download](#coco-dataset-download)
4.  [Contributing](#contributing)
5.  [Notebook] 

## Project Overview

The goal of this notebook is to provide a hands-on demonstration of different machine learning techniques and libraries within the Google Colab environment. It serves as a collection of examples for common tasks in computer vision and natural language processing.

## Setup and Running the Notebook

You can run this notebook in Google Colab or in a local Jupyter Notebook environment.

1.  **Using Google Colab:** The easiest way is to open the `.ipynb` file directly in Google Colab.
2.  **Using Local Jupyter Notebook:**
    *   Clone this repository to your local machine:
    *  Navigate to the cloned directory in your terminal.
    *   Start a Jupyter Notebook or JupyterLab server:
  
      *   Open the `.ipynb` file in your browser.

### Prerequisites

*   **Google Colab:** No specific local setup is required, as Colab provides the necessary environment.
*   **Local Jupyter Notebook:**
    *   Python 3.6+
    *   Install the required libraries using pip. You can run the following command:

 Note that some library installations (specifically for TensorFlow and related packages) are also included within the notebook cells.

## Code Sections

Below is a summary of the different experiments included in the notebook.

### YOLOv4 Object Detection

This section demonstrates how to utilize the Darknet framework (specifically a fork for YOLOv4) to perform object detection. It involves:
*   Cloning the Darknet repository.
*   Compiling the Darknet executable.
*   Downloading pre-trained YOLOv4 weights.
*   Running the detection command on an image and displaying the result.

**Important:** This section is designed to be run in a Colab runtime with GPU acceleration enabled, as compiling Darknet and running the model are computationally intensive. You will need to replace placeholder paths in the code with the actual paths to your `.data` file, `.cfg` file, weights file, and input image.

### Iris Dataset Classification

A classic machine learning example. This section shows how to:
*   Load the Iris dataset directly from a URL into a pandas DataFrame.
*   Separate features (X) and target (Y).
*   Encode the categorical target variable using `LabelEncoder` and then one-hot encode it using `to_categorical`.
*   Prepare the data for training a classification model.

### MNIST Handwritten Digit Recognition

This part focuses on classifying handwritten digits from the MNIST dataset. The steps include:
*   Loading the MNIST training and test datasets using `tf.keras.datasets`.
*   Normalizing the pixel values to a range between 0 and 1.
*   Reshaping the image data to a flat vector suitable for a fully connected neural network.
*   One-hot encoding the labels.
*   Defining and compiling a simple sequential neural network model using TensorFlow/Keras.
*   Training the model and evaluating its performance on the test set.

### Fashion MNIST Image Classification

Similar to the MNIST example, but using the Fashion MNIST dataset which is a slightly more challenging dataset for image classification. This section builds a neural network with four hidden layers and demonstrates:
*   Loading and preprocessing the Fashion MNIST data.
*   Building a deeper sequential model.
*   Compiling the model with specified learning rate, loss function, and additional metrics like Mean Squared Error (MSE).
*   Training the model for an extended number of epochs (100).
*   Plotting the training and validation accuracy over epochs to visualize the learning process.

### Text Generation with RNNs

This experiment delves into basic natural language processing by implementing a character-level text generation model using Recurrent Neural Networks. Key components include:
*   Loading text data from a file (`belling_the_cat.txt`). You will need to create this file with some text content.
*   Building a vocabulary of unique words from the text.
*   Defining a simple RNN model using TensorFlow/Keras layers (Embedding and SimpleRNN).
*   Implementing a training loop that feeds sequences of words to the RNN to predict the next word.
*   Providing an interactive loop to generate text by giving the model a starting sequence of words.

### COCO Dataset Download

A utility section demonstrating how to easily download and extract a large public dataset. This part uses `tf.keras.utils.get_file` to download the COCO 2017 training dataset from a given URL.

## Contributing

Contributions are welcome! If you find any issues or have improvements, feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## Notebook
https://colab.research.google.com/drive/12nEZssycFp_I87bPmS6YFwqW1sEtZIdy?usp=drive_link




https://colab.research.google.com/drive/1EACO0FGWonlacyQ3yMTWAebdFl_MKNKh?usp=sharing
