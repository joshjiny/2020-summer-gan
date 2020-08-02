# Image Generation with Keras Conditional GAN
# Dataset: CIFAR-10
# July 16, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


### 1. IMPORT PACKAGES
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.layers import Input, Dense, Flatten
from keras.layers import Reshape, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
from time import time
import shutil
from keras.preprocessing import image


### 2. CONSTANTS AND HYPER-PARAMETERS
MY_EPOCH = 2
MY_BATCH = 256
MY_NOISE = 100

tags = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog',
        'Frog', 'Horse', 'Ship', 'Truck']

# 1 : we do training from scratch
# 0:  we do prediction using the saved models
TRAINING = 1

# create output directory
OUT_DIR = "./output/"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
else:
    os.makedirs(OUT_DIR)


### 3. LOAD AND MANIPULATE DATASET
# get CIFAR10 dataset
def read_dataset():
    (X_train, Y_train), (_, _) = cifar10.load_data()

    # normalize data
    X_train = (X_train - 127.5) / 127.5

    # 1-hot encode labels
    Y_train = to_categorical(Y_train, 10)

    # print shape info
    print('\n=== DATASET SHAPE INFO')
    print('X_train shape = ', X_train.shape)
    print('Y_train shape = ', Y_train.shape)

    return X_train, Y_train


### 4. MODEL CONSTRUCTION
# build generator using a CNN
# note that we do not use keras sequential model
# input: noise vector (100 values) + conditional input (10 values)
# output: 32 x 32 x 3 fake image
def build_gen(input_layer, condition_layer):
    merged = Concatenate()([input_layer, condition_layer])

    # merge 110 input values and expands to 8,192 values
    hid = Dense(128 * 8 * 8, activation='relu')(merged)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # reshape 8,192 to 8 x 8 x 128 to enter convolution
    hid = Reshape((8, 8, 128))(hid)

    # first convolution block
    # shape remains the same: 8 x 8 x 128
    hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # first convolution transpose block
    # shape expands to 16 x 16 x 128
    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # second convolution block
    # shape remains the same: 16 x 16 x 128
    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # second convolution transpose block
    # shape expands to 32 x 32 x 128
    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # third convolution block
    # shape remains the same: 32 x 32 x 128
    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # fourth convolution block
    # shape changes tp 32 x 32 x 3
    hid = Conv2D(3, kernel_size=5, strides=1, padding="same")(hid)
    out = Activation("tanh")(hid)

    # build the overall model
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    print('\n=== GENERATOR SUMMARY')
    model.summary()

    return model

# build discriminator using a CNN
# note that we do not use keras sequential model
# input: 32 x 32 x 3 image + conditional input (10 values)
# output: 1 sigmoid output (probability)
def build_disc(input_layer, condition_layer):
    # first convolution block
    # shape expands to 32 x 32 x 128
    hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # second convolution block
    # shape shrinks to 16 x 16 x 128
    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # third convolution block
    # shape shrinks to 8 x 8 x 128
    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # fourth convolution block
    # shape shrinks to 4 x 4 x 128
    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)

    # flatten to merge with conditional input
    # 4 x 4 x 128 becomes 2,048 value vector
    hid = Flatten()(hid)

    # merge with 10 conditional input values here
    # 2,048 becomes 2,058
    merged = Concatenate()([hid, condition_layer])

    # two dense layers to obtain the single output
    hid = Dense(512, activation='relu')(merged)
    out = Dense(1, activation='sigmoid')(hid)

    # build the overall model
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    print('\n=== DISCRIMINATOR SUMMARY')
    model.summary()

    return model

# our GAN merges generator and discriminator
def build_GAN():
    # build discriminator first
    img_input = Input(shape=(32, 32, 3))
    disc_cond = Input(shape=(10,))
    discriminator = build_disc(img_input, disc_cond)
    discriminator.compile(optimizer=Adam(0.001, 0.5), loss='binary_crossentropy',
                          metrics=['accuracy'])

    # fix discriminator weights durng generator training
    discriminator.trainable = False

    # build generator. we do NOT compile generator
    noise_input = Input(shape=(MY_NOISE,))
    gen_cond = Input(shape=(10,))
    generator = build_gen(noise_input, gen_cond)

    # combine discriminator and generator to build GAN
    noise = Input(shape=(MY_NOISE,))
    fake = generator([noise, gen_cond])
    gan_out = discriminator([fake, disc_cond])
    cnd_gan = Model(inputs=[noise, gen_cond, disc_cond], output=gan_out)
    cnd_gan.compile(optimizer=Adam(0.001, 0.5), loss='binary_crossentropy')

    print('\n=== GAN SUMMARY ===')
    cnd_gan.summary()

    return generator, discriminator, cnd_gan


### 5. MODEL TRAINING
# generate a batch of random noise vectors
# to be used by generator
def get_noise(n_samples, noise_dim):
      X = np.random.normal(0, 1, size=(n_samples, noise_dim))
      return X

# generate a batch of random labels
# to be used by generator
def rand_label(n):
    y = np.random.choice(10, n)
    y = to_categorical(y, 10)
    return y

# train discriminator
# once with real image/label and next with fake image/label
def train_disc(bid):
    # next batch of real images
    images = X_train[bid * MY_BATCH: (bid + 1) * MY_BATCH]
    labels = Y_train[bid * MY_BATCH: (bid + 1) * MY_BATCH]

    # train discriminator on real data
    # returns two values: loss and accuracy
    all_1 = 1 - np.random.rand(MY_BATCH, 1) * 0.05
    d_loss_real = discriminator.train_on_batch([images, labels], all_1)

    # noise inputs
    noise = get_noise(MY_BATCH, MY_NOISE)

    # generate fake images using true labels
    # discriminator must identify fake "correctly"
    fakes = generator.predict([noise, labels])

    # train discriminator on fake data
    # returns two values: loss and accuracy
    all_0 = np.random.rand(MY_BATCH, 1) * 0.05
    d_loss_fake = discriminator.train_on_batch([fakes, labels], all_0)

    # calculate discriminator loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss[0]

# train generator
def train_gen():
    # generate a batch of fake images
    # we use random noise and random labels
    noise = get_noise(MY_BATCH, MY_NOISE)
    labels_ran = rand_label(MY_BATCH)

    # the goal is to fool discriminator
    # returns one value: loss
    all_1 = 1 - np.random.rand(MY_BATCH, 1) * 0.05
    g_loss = cnd_gan.train_on_batch([noise, labels_ran,
                                     labels_ran], all_1)

    return g_loss

# test generator and display fake images
# we give one-hot conditional input to generator
def save_samples(bid):
    # display 10 images of given category
    fig, axs = plt.subplots(2, 5, figsize=(10,6))

    for tag in range(10):
        # decide which row and column to display
        row = int(tag / 5)
        col = (tag if tag <= 4 else tag - 5)

        label = to_categorical([tag], 10)
        noise = get_noise(1, MY_NOISE)
        gen_img = generator.predict([noise, label])

        # generator returns (1, 32, 32, 3)
        # but array_to_img needs (32, 32, 3)
        img = image.array_to_img(gen_img[0], scale=True)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
        axs[row, col].set_title(tags[tag])

    path = os.path.join(OUT_DIR, "img-{}".format(bid))
    plt.savefig(path)
    plt.close()

# run epochs to train GAN
def train_GAN():
    print('\n=== GAN TRAINING BEGINS\n')

    # total number of batches used for each epoch
    num_batches = int(X_train.shape[0] / MY_BATCH)
    print('Batch count = ', num_batches)

    began = time()
    for epoch in range(MY_EPOCH):
        print('\nEpoch', epoch)
        for bid in range(num_batches):
            # train discriminator and generator
            d_loss = train_disc(bid)
            g_loss = train_gen()

            # print the current status
            print('Batch: {}/{}, generator loss: {:.3f}, '
                  'discriminator loss: {:.3f}'
                  .format(bid, num_batches, g_loss, d_loss))
        save_samples(epoch)

    total = time() - began
    print('\n=== Training Time: = {:.0f} secs, {:.1f} hrs'
          .format(total, total / 3600))


### 6. MODEL EVALUATION
# show sample 9 images per class
def pred_GAN():
    print('\n=== GAN PREDICTION BEGINS')

    # build generator for prediction
    noise_input = Input(shape=(MY_NOISE,))
    gen_cond = Input(shape=(10,))
    generator = build_gen(noise_input, gen_cond)
    generator.load_weights('./generator.h5')

    for tag in range(10):
        # generate 9 1-hot labels and noise vectors for each class
        label = to_categorical([tag] * 9, 10)
        noise = get_noise(9, MY_NOISE)

        # use generator to produce 9 fake images for each class
        fake = generator.predict([noise, label])

        fig, axs = plt.subplots(3, 3)
        count = 0
        for i in range(3):
            for j in range(3):
                # converts 3D numpy array to PIL image instance
                img = image.array_to_img(fake[count], scale=True)
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                plt.suptitle('Label: ' + str(tag) + ', ' +
                             tags[tag])
                count += 1

        # save the images
        path = os.path.join(OUT_DIR, "Final {}".format(tags[tag]))
        plt.savefig(path)
        plt.close()


### 7. MAIN ROUTINE
# if training is not done, we do training and save the models
# otherwise, we use the saved models to do prediction
X_train, Y_train = read_dataset()

if TRAINING:
    generator, discriminator, cnd_gan = build_GAN()
    train_GAN()
    generator.save_weights(os.path.join(OUT_DIR, 'generator.h5'))
else:
    pred_GAN()
