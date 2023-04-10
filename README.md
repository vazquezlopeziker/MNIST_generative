# MNIST_generative
A generative model based on MNIST dataset that can produce number images.

## Requirements
* Pytorch
* OpenCV (for interactive generation)

## Training
Use the config.py file to set up the directory structure and the preferred training configuration.

    >cd MNIST_generative
    >python src/main.py -train

## Test functionality
To test that the model learnt from the dataset and to check some results use:

    >cd MNIST_generative
    >python src/main.py

To change the testing image, replace the index in line main.py:34 for the desired index.

    img, _ = train_data[0] # replace the 0

## Running interactive MNIST generation
I created an interative system to draw on an image and the autoencoder will generate a new image based on the drawn input. The interaction shows how the network generates the images and, since it contains noise, it is possible to visualize the predictions on how the number will look like.

To run the interactive generation use:

    >cd MNIST_generative
    >python src/interactive_MNIST_generation.py