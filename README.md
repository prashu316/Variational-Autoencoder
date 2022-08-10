# Variational-Autoencoder

## Aim
To generate new images not present in a dataset using random latent space coordinates. This is done with the help of a variational autoencoder which maps inputs to a normal distribution belonging to a comparatively smaller dimension than the input dimension.

## Libraries Used
Keras

## Dataset Details
The dataset used to generate new images is the MNIST handwritten digits dataset consisting of handwritten digits in grayscale format.

## Outputs

A visualisation of the latent space after training: 

![latent space](https://user-images.githubusercontent.com/57295909/183833572-f989be3c-d285-4ab6-a65e-2519debac84b.png)

Newly generated digits from random latent space coordinates:
![generated digits](https://user-images.githubusercontent.com/57295909/183833601-cf88badc-18aa-4662-8b6a-c14be2c4f412.png)

Notebook along with code can be found in this repository or you could click [here](https://github.com/prashu316/Variational-Autoencoder/blob/main/VAE.ipynb)
