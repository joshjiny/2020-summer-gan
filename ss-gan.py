# MNIST Hand-written Digit Generation with Keras Semi-Supervised GAN
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
from keras.layers import Activation, BatchNormalization, Dense
from keras.layers import Dropout, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.utils import to_categorical


### 2. CONSTANTS AND HYPER-PARAMETERS
MY_EPOCH = 20
MY_BATCH = 1000
MY_NOISE = 100
img_shape = (28, 28, 1)
num_classes = 10

# number of labeled examples to use
# rest will be used as unlabeled
num_labeled = 100

# create output directory
OUT_DIR = "./output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


### 3. LOAD AND MANIPULATE DATASET
# read and pre-process MNIST dataset
def read_dataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    Y_train = Y_train.reshape(60000, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    Y_test = Y_test.reshape(10000, 1)

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    return X_train, Y_train, X_test, Y_test

# random batch from the first 100 training images
# this is used to train supervised discriminator
def batch_labeled():
    idx = np.random.randint(0, num_labeled, MY_BATCH)
    imgs = X_train[idx]
    labels = Y_train[idx]

    return imgs, labels

# random batch from the remaining 59,900 training images
# this is used to train unsupervised discriminator
def batch_unlabeled():
    idx = np.random.randint(num_labeled, X_train.shape[0],
                            MY_BATCH)
    imgs = X_train[idx]

    return imgs


### 4. MODEL CONSTRUCTION
# build CNN-based generator
# input: one dimensional noise vector
# output: 28 x 28 x 1 fake image
def build_generator():
    model = Sequential()

    # reshape noise input into 7x7x256 tensor via dense layer
    model.add(Dense(7 * 7 * 256, input_dim=MY_NOISE))
    model.add(Reshape((7, 7, 256)))

    # 1st transposed convolution block
    # dimension becomes 14 x 14 x 128
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 2nd transposed convolution block
    # dimension becomes 14 x 14 x 64
    # image size stays the same
    model.add(Conv2DTranspose(64, kernel_size=3, strides = 1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 3rd transposed convolution block
    # dimension becomes 28 x 28 x 1
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    # output layer with tanh activation
    model.add(Activation('tanh'))

    # print the summary of generator
    print('\n=== Generator summary')
    model.summary()

    return model


# build CNN-based discriminator (common part)
# input: 28 x 28 x 1 image (real or fake) 
# output: 10 dense neuron outputs
def build_disc_common():
    model = Sequential()

    # 1st convolutional block
    # dimension becomes 14 x 14 x 32
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape = img_shape,
                     padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    # 2nd convolutional block
    # dimension becomes 7 x 7 x 64
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # 3rd convolutional block
    # dimension becomes 3 x 3 x 128
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))

    # flatten the tensor
    model.add(Flatten())

    # fully connected layer with num_classes neurons
    model.add(Dense(10))

    # print the summary of discriminator
    print('\n=== Discriminator (common) summary')
    model.summary()

    return model

# add softmax activation at the end of discriminator
# giving predicted probability distribution over the real classes
# input: 10 values
# output: 10 softmax probability values
def build_disc_super(common):
    model = Sequential()
    model.add(common)
    model.add(Activation('softmax'))

    return model


# add sigmoid activation at the end of discriminator
# to decide real vs. fake 
# input: 10 values
# output: single probability value 
def build_disc_unsuper(common):
    model = Sequential()
    model.add(common)
    model.add(Dense(1, activation = 'sigmoid'))

    return model


# build three discriminators: common, supervised, unsupervised
# our GAN merges generator and unsupervised discriminator
def build_GAN():
    # build the common discriminator
    # shared between supervised and unsupervised training
    share = build_disc_common()

    # build & compile the discriminator for supervised training
    disc_super = build_disc_super(share)
    disc_super.compile(optimizer='Adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # build & compile the discriminator for unsupervised training
    # note that we do not care about accuracy
    disc_unsuper = build_disc_unsuper(share)
    disc_unsuper.compile(optimizer='Adam',
                         loss='binary_crossentropy')

    # build generator
    generator = build_generator()

    # fix discriminator during generator training
    disc_unsuper.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(disc_unsuper)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    print('\n=== GAN summary')
    model.summary()

    return generator, disc_unsuper, disc_super, model


### 5. MODEL TRAINING
# we train discriminator 3 times
# 1. supervised training with real images and true labels
# 2. unsupervised training with real images and all-ones
# 3. unsupervised training with fake images and all-zeros
def train_discriminator():
    # get labeled examples
    imgs, labels = batch_labeled()

    # supervised train on real labeled examples
    # supervised discriminator has 10 softmax outputs
    # returns two values: loss and accuracy
    d_loss_super, _ = disc_super.train_on_batch(imgs, labels)

    # labels for real and fake images
    all_1 = np.ones((MY_BATCH, 1))
    all_0 = np.zeros((MY_BATCH, 1))

    # get unlabeled examples
    imgs_unlabeled = batch_unlabeled()

    # generate a batch of fake images
    z = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))
    fake = generator.predict(z)

    # unsupervised train on real unlabeled examples
    # unsupervised discriminator has 1 sigmoid output
    # returns one value: loss
    d_loss_real = disc_unsuper.train_on_batch(imgs_unlabeled,
                                              all_1)

    # unsupervised train on fake examples
    # returns one value: loss
    d_loss_fake = disc_unsuper.train_on_batch(fake, all_0)

    # computing loss by taking the average
    d_loss_unsuper = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss_super, d_loss_unsuper


# train generator
def train_generator():
    # generate a batch of fake images
    z = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))

    # note we train the entire gan but fix discriminator weights
    # returns one value: loss
    all_1 = np.ones((MY_BATCH, 1))
    g_loss = gan.train_on_batch(z, all_1)

    return g_loss

# test generator with sample images
def save_images(itr):
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

# GAN training function
def train_GAN():
    print('\n=== GAN TRAINING BEGINS\n')

    # repeat epochs
    began = time()
    for itr in range(MY_EPOCH):
        d_loss_s, d_loss_u = train_discriminator()
        g_loss = train_generator()

        # output generator and discriminator loss values
        if itr % 100 == 0 or True:
            print('Epoch: {}, generator loss: {:.3f}, '
                  'discriminator super loss: {:.3f}, '
                  'unsuper loss: {:.3f}'
                  .format(itr, g_loss, d_loss_s, d_loss_u))

            # output a sample of generated image
            save_images(itr)

    # print training time
    total = time() - began
    print('\n=== Training Time: = {:.0f} secs, {:.1f} hrs'
          .format(total, total / 3600))



### 6. MODEL EVALUATION
#  first 100 images used to train supervised discriminator
def training_set():
    imgs = X_train[range(num_labeled)]
    labels = Y_train[range(num_labeled)]

    return imgs, labels

# evaluating semi-supervised discriminator
# compute classification accuracy on training set
def eval_disc():
    print('\n=== SUPERVISED DISCRIMINATOR EVALUATION')
    x, y = training_set()
    _, accuracy = disc_super.evaluate(x, y, verbose=0)
    print("Training accuracy: %.2f%%" % (100 * accuracy))

    # evaluating semi-supervised discriminator
    # compute classification accuracy on test set
    _, accuracy = disc_super.evaluate(X_test, Y_test, verbose=0)
    print("Test accuracy: %.2f%%" % (100 * accuracy))


### 7. MAIN ROUTINE
# read MNIST dataset and obtain all 4 sets
X_train, Y_train, X_test, Y_test = read_dataset()

# build and compile GAN
generator, disc_unsuper, disc_super, gan = build_GAN()

# train GAN and report training time
train_GAN()

# evaluating semi-supervised discriminator
eval_disc()