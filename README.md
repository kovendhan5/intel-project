# Pixelated Image Correction using Machine Learning

This project aims to correct pixelated images by utilizing a Convolutional Neural Network (CNN) model. The project consists of two main parts: training the model and using the trained model to upscale and correct pixelated images.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Upscaling Pixelated Images](#upscaling-pixelated-images)
- [Results](#results)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Pixelated images are common in various scenarios, such as low-resolution cameras or privacy-preserving image sharing. This project addresses the challenge of restoring high-quality images from pixelated versions by leveraging deep learning techniques.

## Project Structure

├── train_model.py # Code for training the model
├── upscaling.py # Code for upscaling and correcting pixelated images
├── README.md # Project documentation
├── /content/train # Directory containing training images
├── /content/test # Directory containing test images
├── /content/Image/Pixelated # Directory containing pixelated images for upscaling
├── /content/Image/Output # Directory for storing output images


## Dependencies

To run this project, you will need the following dependencies:

- Python 3.x
- TensorFlow
- NumPy
- Pillow
- Matplotlib
- scikit-learn

You can install the dependencies using pip:

```bash
pip install tensorflow numpy pillow matplotlib scikit-learn


**Dataset**
The dataset should consist of two main directories:

/content/train: Contains training images with subdirectories for each class (e.g., Original and Pixelated).
/content/test: Contains test images with subdirectories for each class.
Ensure that the dataset is organized properly for the ImageDataGenerator to work correctly.

Model Training
The train_model.py script is used to train the CNN model. The script performs the following steps:

Defines image dimensions and batch size.
Creates image data generators for training and testing datasets.
Builds a lightweight CNN model for binary classification (original vs. pixelated).
Compiles and trains the model.
Evaluates the model and prints the test accuracy.
Saves the trained model and quantizes it for better performance.
Usage
Run the following command to train the model:

bash
Copy code
python train_model.py
Upscaling Pixelated Images
The upscaling.py script is used to upscale and correct pixelated images using the trained and quantized TFLite model. The script performs the following steps:

Preprocesses the input image.
Loads the TFLite model and allocates tensors.
Invokes the model to process the image.
Saves and plots the super-resolution image.
Usage
Run the following command to upscale pixelated images:

bash
python upscaling.py

Results
The model is evaluated using test accuracy, F1 Score, Precision, Recall, and Confusion Matrix. The quantized TFLite model size is also displayed for performance comparison.

How to Use
Prepare the dataset and place it in the appropriate directories (/content/train and /content/test).
Train the model using train_model.py.
Place the pixelated images to be corrected in /content/Image/Pixelated.
Run the upscaling.py script to generate the corrected images in /content/Image/Output.

Contributing
Contributions are welcome! If you have any ideas, improvements, or bug fixes, feel free to create a pull request or open an issue.


