# MNIST Hand-written Digit Generation with Keras Deep Convolutional GAN
# Dataset: MNIST
# July 14, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


### 1. IMPORT PACKAGES
import matplotlib.pyplot as plt
import numpy as np
import os

from time import time
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential


### 2. CONSTANTS AND HYPER-PARAMETERS
MY_EPOCH = 20
MY_BATCH = 128
MY_NOISE = 100
img_shape = (28, 28, 1)

# create output directory
OUT_DIR = "./output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


### 3. LOAD AND MANIPULATE DATASET
def read_dataset():

    # we only need train data input (= unsupervised learning)
    (X_train, _), (_, _) = mnist.load_data()
    print('\n=== Train input set shape:', X_train.shape)

    # rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0

    # add channel info
    X_train = np.expand_dims(X_train, axis=3)
    print('=== Train input set after reshaping:', X_train.shape)

    return X_train


### 4. MODEL CONSTRUCTION
# deep convolution generator
# input: noise input vector (100 values)
# output: 28 x 28 x 1 fake image
def build_generator():
    model = Sequential()

    # reshape input into 7 x 7 x 256 tensor via a fully connected layer
    # convolution needs 3-dimension input
    # note that reshape needs (())
    model.add(Dense(7 * 7 * 256, input_dim=MY_NOISE))
    model.add(Reshape((7, 7, 256)))

    # 1st transposed convolution layer, from 7 x 7x 256 into 14 x 14 x 128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 2nd transposed convolution layer, from 14 x 14 x 128 to 14 x 14 x 64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 3rd transposed convolution layer, from 14 x 14 x 64 to 28 x 28 x 1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    # output layer with tanh activation
    model.add(Activation('tanh'))

    # print model summary
    print('\n=== Generator summary')
    model.summary()

    return model


# deep convolution discriminator
# input: 28 x 28 x 1 
# output: 1 value (real or fake)
def build_discriminator():
    model = Sequential()

    # 1st convolutional layer, from 28 x 28 x 1 into 14 x 14 x 32 tensor
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape = img_shape, 
        padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    # 2nd convolutional layer, from 14 x 14 x 32 into 7 x 7 x 64 tensor
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 3rd convolutional layer, from 7 x 7 x 64 tensor into 3 x 3 x 128 tensor
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # print model summary
    print('\n=== Discriminator summary')
    model.summary()

    return model

# build GAN by merging generator and discriminator
def build_GAN():
    model = Sequential()

    # build discriminator first
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer='Adam',
                          metrics=['accuracy'])

    # build generator next
    # we do not compile generator separately
    generator = build_generator()

    # keep discriminator’s parameters constant for generator training
    discriminator.trainable = False

    # merge generator and discriminator
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='Adam')

    print('\n=== GAN summary')
    model.summary()

    return discriminator, generator, model


### 5. MODEL TRAINING
# train discriminator
def train_discriminator():
    # label for real images: all ones
    # double parenthesis are needed to create 2-dim array
    all_1 = np.ones((MY_BATCH, 1))

    # label for fake images: all zeros
    all_0 = np.zeros((MY_BATCH, 1))

    # get a random batch of real images
    idx = np.random.randint(0, X_train.shape[0], MY_BATCH)
    real = X_train[idx]

    # generate a batch of fake images
    z = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))

    # use generator to produce fake images
    fake = generator.predict(z)

    # discriminator weights change, but generator weights are untouched
    # returns two values: loss and accuracy
    d_loss_real = discriminator.train_on_batch(real, all_1)
    d_loss_fake = discriminator.train_on_batch(fake, all_0)
    d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss, accuracy

# train generator
def train_generator():
    # generate a batch of fake images
    z = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))

    # note we train the entire gan but fix discriminator weights
    # returns one value: loss
    all_1 = np.ones((MY_BATCH, 1))
    g_loss = gan.train_on_batch(z, all_1)

    return g_loss

# GAN training function
def train_GAN():
    print('\n=== GAN TRAINING BEGINS\n')

    # repeat epochs
    began = time()
    for itr in range(MY_EPOCH):
        d_loss, acc = train_discriminator()
        g_loss = train_generator()

        # test generator and save training loss info
        if itr % 100 == 0 or True:
            # output generator and discriminator loss values
            if itr % 100 == 0 or True:
                print('Epoch: {}, generator loss: {:.3f}, discriminator loss: {:.3f}, '
                      'accuracy: {:.1f}%'.format(itr, g_loss, d_loss, acc * 100))

            # output a sample of generated image
            sample_images(itr)

    # print training time
    total = time() - began
    print('\n=== Training Time: = {:.0f} secs, {:.1f} hrs'.format(total, total / 3600))

# we test generator by showing 16 sample images
def sample_images(itr):
    # display 16 images
    row = col = 4

    # generate 16 fake images from random noise
    z = np.random.normal(0, 1, (row * col, MY_NOISE))
    fake = generator.predict(z)

    # rescale image pixel values from [-1, 1] to [0, 1]
    fake = 0.5 * fake + 0.5
    _, axs = plt.subplots(row, col, figsize=(row, col))

    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(fake[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1

    path = os.path.join(OUT_DIR, "img-{}".format(itr + 1))
    plt.savefig(path)
    plt.close()


### 6. MAIN ROUTINE
# read MNIST dataset, only the training input set
X_train = read_dataset()

# build and compile GAN
discriminator, generator, gan = build_GAN()

# train GAN and report training time
train_GAN()


