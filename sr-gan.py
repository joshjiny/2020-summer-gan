# Image Resolution Enhancement with Keras Super Resolution GAN
# Dataset: Celebrity Photos
# July 18, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


### 1. IMPORT PACKAGES
import os, glob
import matplotlib.pyplot as plt
import numpy as np

from keras import Input
from time import time
from keras.applications import ResNet50
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, Add, Dense
from keras.layers import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from imageio import imread
from skimage.transform import resize


### 2. CONSTANTS AND HYPER-PARAMETERS
MY_EPOCH = 5
MY_BATCH = 2
LOW_SHAPE = (64, 64, 3)
HIGH_SHAPE = (256, 256, 3)
DIS_ANS = (MY_BATCH, 16, 16, 1)

MY_MOM = 0.8
MY_ALPHA = 0.2
MY_RESIDUAL = 15
MY_OPT = Adam(0.0002, 0.5)

# 1 : we do training from scratch
# 0:  we do prediction using the saved models
TRAINING = 1

# directories
DB_DIR = "./celeb/*.*"
OUT_DIR = "./output"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


### 3. LOAD AND MANIPULATE DATASET
# get a random batch of images from the dataset
# image shape is 218 x 178 x 3 (RGB color)
def sample_images():
    # choose a random batch of images
    all_images = glob.glob(DB_DIR)
    images = np.random.choice(all_images, size=MY_BATCH)

    low_img = []
    high_img = []
    for img in images:
        # get numpy array of the current image
        new_img = imread(img, pilmode='RGB')
        new_img = new_img.astype(np.float32)

        # resize images
        img_high = resize(new_img, HIGH_SHAPE)
        img_low = resize(new_img, LOW_SHAPE)

        # add to the final list
        high_img.append(img_high)
        low_img.append(img_low)

    # convert to numpy arrays
    high_img = np.array(high_img)
    high_img = np.array(high_img)

    # normalize the pixel values to [-1, 1]
    # this works well with tanh activation
    high_img = np.subtract(np.divide(high_img, 127.5), 1)
    low_img = np.subtract(np.divide(low_img, 127.5), 1)

    return high_img, low_img


### 4. MODEL CONSTRUCTION
# use pre-trained resnet50 model
# input: 256 x 256 x 3 image
# output: 1000 features
def build_resnet():
    # use keras resnet50 utility
    resnet = ResNet50(weights='imagenet')
    input_L = Input(shape=HIGH_SHAPE)

    # resnet50 accepts 224 x 224 x 3
    # dense layer between 256 x 256 x 3 and 224 x 224 x 3
    features = resnet(input_L)

    # create a keras model
    # we do not train resnet50 further
    model = Model(inputs=[input_L], outputs=[features])
    model.trainable = False

    print('\n=== RESNET50 SUMMARY')
    model.summary()

    return model

# common convolution definition
def MY_CONV(channel, kernel, stride):
    return Conv2D(channel, kernel, strides=stride, padding='SAME')

# discriminator network using deep CNN
# input: 256 x 256 x 3 high resolution image
# output: 16 x 16 x 1 sigmoid probabilities
def build_discriminator():
    input_L = Input(shape=HIGH_SHAPE)

    # first convolution block
    dis1 = MY_CONV(64, 3, 1)(input_L)
    dis1 = LeakyReLU(MY_ALPHA)(dis1)

    # second convolution block
    # image shape reduces to 128 x 128 x 64
    dis2 = MY_CONV(64, 3, 2)(dis1)
    dis2 = LeakyReLU(MY_ALPHA)(dis2)
    dis2 = BatchNormalization(momentum = MY_MOM)(dis2)

    # third convolution block
    dis3 = MY_CONV(128, 3, 1)(dis2)
    dis3 = LeakyReLU(MY_ALPHA)(dis3)
    dis3 = BatchNormalization(momentum = MY_MOM)(dis3)

    # fourth convolution block
    # image shape reduces to 64 x 64 x 128
    dis4 = MY_CONV(128, 3, 2)(dis3)
    dis4 = LeakyReLU(MY_ALPHA)(dis4)
    dis4 = BatchNormalization(momentum = MY_MOM)(dis4)

    # fifth convolution block
    dis5 = MY_CONV(256, 3, 1)(dis4)
    dis5 = LeakyReLU(MY_ALPHA)(dis5)
    dis5 = BatchNormalization(momentum = MY_MOM)(dis5)

    # sixth convolution block
    # image shape reduces to 32 x 32 x 256
    dis6 = MY_CONV(256, 3, 2)(dis5)
    dis6 = LeakyReLU(MY_ALPHA)(dis6)
    dis6 = BatchNormalization(momentum = MY_MOM)(dis6)

    # seventh convolution block
    dis7 = MY_CONV(512, 3, 1)(dis6)
    dis7 = LeakyReLU(MY_ALPHA)(dis7)
    dis7 = BatchNormalization(momentum = MY_MOM)(dis7)

    # eight convolution block
    # image shape reduces to 16 x 16 x 512
    dis8 = MY_CONV(512, 3, 2)(dis7)
    dis8 = LeakyReLU(MY_ALPHA)(dis8)
    dis8 = BatchNormalization(momentum = MY_MOM)(dis8)

    # add a dense layer
    # image channel reduces to 16 x 16 x 1024
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(MY_ALPHA)(dis9)

    # last dense layer with single output for classification
    # image channel reduces to 16 x 16 x 1
    output = Dense(units=1, activation='sigmoid')(dis9)

    # final keras model for discriminator
    model = Model(inputs=[input_L], outputs=[output])

    print('\n== DISCRIMINATOR MODEL SUMMARY ==')
    model.summary()

    return model

# residual block used in generator
def residual_block(x):
    # first convolution block
    res = MY_CONV(64, 3, 1)(x)
    res = Activation(activation="relu")(res)
    res = BatchNormalization(momentum=MY_MOM)(res)

    # second convolution block
    res = MY_CONV(64, 3, 1)(res)
    res = BatchNormalization(momentum=MY_MOM)(res)

    # add bypass synaptic connections
    res = Add()([res, x])

    return res

# generator network using deep CNN
# input: 64 x 64 x 3 low resolution image
# output: 256 x 256 x 3 high resolution image
def build_generator():
    input_L = Input(shape=LOW_SHAPE)

    # pre-residual block
    # channel increases to 64 x 64 x 64
    gen1 = Conv2D(64, 9, strides=1, padding='same', activation='relu')(input_L)

    # add 15 residual blocks
    # shape remains 64 x 64 x 64
    res = residual_block(gen1)
    for i in range(MY_RESIDUAL):
        res = residual_block(res)

    # post-residual block
    # shape remains 64 x 64 x 64
    gen2 = MY_CONV(64, 3, 1)(res)
    gen2 = BatchNormalization(momentum = MY_MOM)(gen2)

    # take the sum of the output from pre-residual block (gen1) 
    # and the post-residual block (gen2)
    gen3 = Add()([gen2, gen1])

    # shape expands to 128 x 128 x 64
    # then to 128 x 128 x 256
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = MY_CONV(256, 3, 1)(gen4)
    gen4 = Activation('relu')(gen4)

    # add another up-sampling block
    # shape expands to 256 x 256 x 256
    gen5 = UpSampling2D(size = 2)(gen4)
    gen5 = MY_CONV(256, 3, 1)(gen5)
    gen5 = Activation('relu')(gen5)

    # output convolution layer
    # channel reduces to 256 x 256 x 3
    gen6 = MY_CONV(3, 9, 1)(gen5)
    output = Activation('tanh')(gen6)

    # keras model of our generator
    model = Model(inputs = [input_L], outputs = [output])
    print('\n=== GENERATOR SUMMARY')
    model.summary()

    return model

# merge resnet50, discriminator, and generator to form GAN
# input: 64 x 64 x 3 low resolution image
# output 1: 16 x 16 probabilities (from discriminator)
# output 2: 1000 features (from RESNET50)
def build_GAN():
    # obtain RESNET50 network to extract features
    resnet = build_resnet()

    # build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss='mse', optimizer=MY_OPT,
                          metrics=['acc'])

    # build generator (we do not compile generator)
    generator = build_generator()

    # generator accepts low resolution images
    # and produce high resolution fake images
    input_low = Input(shape=LOW_SHAPE)
    fake_high = generator(input_low)

    # discriminator accepts fake images and decide
    probs = discriminator(fake_high)

    # resnet50 accepts fake images and decide
    features = resnet(fake_high)

    # discriminator is non-trainable during generator training
    discriminator.trainable = False

    # GAN input: low resolution image (goes to generator)
    # GAN outputs: probs (discriminator) and features (resnet-50)
    sr_gan = Model([input_low], [probs, features])

    # GAN loss = 0.001 x entropy (probs) + 1 x mse (features)
    sr_gan.compile(loss=['binary_crossentropy', 'mse'],
            loss_weights=[0.001, 1], optimizer = MY_OPT)

    print('\n=== GAN SUMMARY')
    sr_gan.summary()

    return resnet, discriminator, generator, sr_gan


### 5. MODEL TRAINING
# train discriminator
# once with real image/label and next with fake image/label
def train_disc():
    # sample a batch of images
    real_high, real_low = sample_images()

    # generate high-resolution images from low-resolution images
    fake_high = generator.predict(real_low)

    # 16 x 16 matrix to represent real and fake labels
    all_1 = np.ones(DIS_ANS)
    all_0 = np.zeros(DIS_ANS)

    # train discriminator using real image
    # returns two values: loss and accuracy
    d_loss_real = discriminator.train_on_batch(real_high, all_1)

    # train discriminator using fake image
    # returns two values: loss and accuracy
    d_loss_fake = discriminator.train_on_batch(fake_high, all_0)

    # average out the two loss values
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    return d_loss[0]

# train generator
# generator needs to fool both discriminator & resnet-50!
def train_gen():
    # sample a new batch of images
    real_high, real_low = sample_images()

    # true label resnet-50 produces
    resnet_1 = resnet.predict(real_high)

    # true label for discriminator
    all_1 = np.ones(DIS_ANS)

    # generator (= SR-GAN) accepts low resolution image
    # returns three values: combined loss, entropy loss, mse loss
    g_loss = sr_gan.train_on_batch([real_low], [all_1, resnet_1])

    return g_loss[0]

# save low-res, high-res (original) and fake high-res images
def save_images(low, original, fake, path):
    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow((low * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow((original * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow((fake * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Fake")

    plt.savefig(path)
    plt.close()

# we pick two random images to test our GAN model
def evaluate_GAN(epoch):
    # sample a new batch of images
    high, low = sample_images()

    # we use generator to turn a low resolution image to high
    fake = generator.predict_on_batch(low)

    # save the images in the current batch
    for i in range(MY_BATCH):
        path = os.path.join(OUT_DIR, "img-{}-{}".format(epoch, i))
        save_images(low[i], high[i], fake[i], path)

# overall GAN training
# we alternate between discriminator and generator training 
def train_GAN():
    print('\n=== GAN TRAINING BEGINS\n')

    # repeat epochs
    began = time()
    for epoch in range(MY_EPOCH):
        # train discriminator and calculate loss
        d_loss = train_disc()

        # train generator and calculate loss
        g_loss = train_gen()

        # print loss
        if (epoch % 50) == 0 or True:
            print("Epoch: {}, discriminator loss: {:.3f}, "
                  "generator loss: {:.3f}"
                  .format(epoch, d_loss, g_loss))
            evaluate_GAN(epoch)

    # print training time
    total = time() - began
    print('\n=== Training Time: = {:.0f} secs, {:.1f} hrs'
          .format(total, total / 3600))


### 6. MODEL EVALUATION
# prediction with GAN
def pred_GAN():
    print('\n=== GAN PREDICTION BEGINS')

    # we just need a trained generator
    generator = build_generator()
    generator.load_weights('./generator.h5')

    # use 5 batches for prediction
    for i in range(5):
        # sample a new batch of images
        high, low = sample_images()

        # we use generator to produce fake high image
        fake = generator.predict_on_batch(low)

        # save the images in the current batch
        for j in range(MY_BATCH):
            path = os.path.join(OUT_DIR, "pred-{}-{}"
                                .format(i, j))
            save_images(low[j], high[j], fake[j], path)


### 7. MAIN ROUTINE
# if training is not done, we do training and save the model
# otherwise, we use the saved model to do prediction

if TRAINING:
    resnet, discriminator, generator, sr_gan = build_GAN()
    train_GAN()
    generator.save_weights(os.path.join(OUT_DIR, "generator.h5"))
else:
    pred_GAN()
