# Turning Paintings into Photos using Keras Cycle-GAN
# Dataset: Monet Paintings and Photos
# July 18, 2020
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


### 1. IMPORT PACKAGES
import matplotlib.pyplot as plt
import numpy as np
import os

from time import time
from glob import glob
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Add, Conv2DTranspose
from keras.layers import ZeroPadding2D, LeakyReLU
from keras.optimizers import Adam
from imageio import imread
from skimage.transform import resize


### 2. CONSTANTS AND HYPER-PARAMETERS
MY_EPOCH = 10
MY_BATCH = 2
in_shape = (128, 128, 3)
out_shape = (MY_BATCH, 7, 7, 1)
residual_blocks = 6
hidden_layers = 3
my_opt = Adam(0.002, 0.5)

# 1 : we do training from scratch
# 0:  we do prediction using the saved models
TRAINING = 1

# create directories
data_dir = "./monet2photo/"
OUT_DIR = "./output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


### 3. LOAD AND MANIPULATE DATASET
# indicate the location of dataset
def loc_dataset():
    # read image files using glob package
    images_A = glob(data_dir + 'trainA/*.*')
    images_B = glob(data_dir + 'trainB/*.*')

    print('\n=== Images in the dataset:')
    print('Train A (paintings):', len(images_A))
    print('Train B (photos):', len(images_B))

    return images_A, images_B

# load image files using glob package
def load_images():
    set_A = []
    set_B = []

    for index, filename in enumerate(images_A):
        imgA = imread(filename, pilmode = 'RGB')
        imgB = imread(images_B[index], pilmode = 'RGB')

        # reduce image resolution: 256 x 256 to 128 x 128
        imgA = resize(imgA, (128, 128))
        imgB = resize(imgB, (128, 128))

        # add images to the final set
        set_A.append(imgA)
        set_B.append(imgB)

    # normalize images to [-1, 1]
    set_A = np.array(set_A) / 127.5 - 1.
    set_B = np.array(set_B) / 127.5 - 1.

    # print the statistics
    print('Paintings used:', len(set_A))
    print('Photos used:', len(set_B))

    return set_A, set_B


### 4. MODEL CONSTRUCTION
# residual block used in the generators
def residual_block(input):
    res = Conv2D(128, kernel_size=3, strides=1, padding='same')(input)
    res = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(res)
    res = Activation('relu')(res)

    res = Conv2D(128, kernel_size=3, strides=1, padding='same')(res)
    res = BatchNormalization(axis=3, momentum = 0.9, epsilon=1e-5)(res)

    return Add()([res, input])

# generator definition. it has auto-encoder shape
# input: 128 x 128 x 3 image
# output: 128 x 128 x 3 image
def build_generator():
    input_layer = Input(shape=in_shape)

    # 1st convolution block
    # shape changes to 128 x 128 x 32
    x = Conv2D(32, kernel_size=7, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 2nd convolution block
    # shape changes to 64 x 64 x 64
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 3rd convolution block
    # shape changes to 32 x 32 x 128
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # residual blocks
    # shape stays at 32 x 32 x 128
    for i in range(residual_blocks):
        x = residual_block(x)

    # 1st upsampling block
    # shape changes to 64 x 64 x 64
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 2nd upsampling block
    # shape changes to 128 x 128 x 32
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # final convolution layer
    # shape changes to 128 x 128 x 3
    x = Conv2D(3, kernel_size=7, strides=1, padding='same')(x)
    output = Activation('tanh')(x)

    # overall model
    model = Model(inputs=[input_layer], outputs=[output])

    return model

# discriminator definition
# input: 128 x 128 x 3 image
# output: 7 x 7 x 1 sigmoid probabilities
def build_discriminator():
    input_layer = Input(shape=in_shape)

    # add 1-layer zero padding around the image
    # shape changes to 130 x 130 x 3
    x = ZeroPadding2D(padding=(1, 1))(input_layer)

    # 1st convolutional block
    # shape changes to 64 x 64 x 64
    x = Conv2D(64, kernel_size=4, strides=2, padding='valid')(x)
    x = LeakyReLU(alpha = 0.2)(x)

    # shape changes to 66 x 66 x 64
    x = ZeroPadding2D(padding=(1, 1))(x)

    # 3 hidden convolution blocks
    # shape changes to 32 x 32 x 128
    # shape changes to 16 x 16 x 256
    # shape changes to 8 x 8 x 256
    # final shape is 10 x 10 x 512
    for i in range(1, hidden_layers + 1):
        x = Conv2D(2 ** i * 64, kernel_size=4, strides=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = ZeroPadding2D(padding = (1, 1))(x)

    # final convolution layer
    # shape changes to 7 x 7 x 1
    output = Conv2D(1, kernel_size=4, strides=1, activation="sigmoid")(x)

    # overall model
    model = Model(inputs=[input_layer], outputs=[output])

    return model

# build two discriminators
# A is for painting, B is for photo
def build_two_disc():
    # discriminator A
    discriminatorA = build_discriminator()    
    print('\n=== Discriminator A summary')
    discriminatorA.summary()

    # discriminator B
    discriminatorB = build_discriminator()
    print('\n=== Discriminator B summary')
    discriminatorB.summary()

    discriminatorA.compile(loss='mse', optimizer=my_opt, metrics=['accuracy'])
    discriminatorB.compile(loss='mse', optimizer=my_opt, metrics=['accuracy'])
    
    return discriminatorA, discriminatorB


# build two generators
# A is for painting, B is for photo
def build_two_gen():
    # generator A to B
    generatorAtoB = build_generator()
    print('\n=== Generator A to B summary')
    generatorAtoB.summary()

    # generator B to A 
    generatorBtoA = build_generator()
    print('\n=== Generator B to A summary')
    generatorBtoA.summary()

    return generatorAtoB, generatorBtoA

# build cycle-GAN
# input: two original images (one painting, one photo)
# output: two probabilities and two reconstructed images
def build_GAN():
    # first flow of data for painting
    input_A = Input(shape=in_shape)
    gen_B = generatorAtoB(input_A)
    prob_B = discriminatorB(gen_B)
    recon_A = generatorBtoA(gen_B)

    # second flow of data for photo
    input_B = Input(shape=in_shape)
    gen_A = generatorBtoA(input_B)
    prob_A = discriminatorA(gen_A)
    recon_B = generatorAtoB(gen_A)

    # make both discriminators non-trainable
    discriminatorA.trainable = False
    discriminatorB.trainable = False

    # build and compile cycle-GAN
    cy_gan = Model(inputs=[input_A, input_B],
        outputs=[prob_A, prob_B, recon_A, recon_B])

    # compile cycle-GAN based on combined loss function
    cy_gan.compile(loss=['mse', 'mse', 'mae', 'mae'],
        loss_weights=[1.0, 1.0, 10.0, 10.0], optimizer = my_opt)

    print('\n=== Cycle-GAN summary')
    cy_gan.summary()

    return cy_gan   

# pick the next batch of random images
# to test GAN and save the resulting images
def load_test_batch():
    pick_A = np.random.choice(images_A, MY_BATCH)
    pick_B = np.random.choice(images_B, MY_BATCH)

    all_A = []
    all_B = []
    for i in range(MY_BATCH):
        # load and resize images
        imgA = resize(imread(pick_A[i], pilmode = 'RGB').astype(np.float32), (128, 128))
        imgB = resize(imread(pick_B[i], pilmode = 'RGB').astype(np.float32), (128, 128))

        all_A.append(imgA)
        all_B.append(imgB)

    # normalize images to [-1, 1]
    all_A = np.array(all_A) / 127.5 - 1.
    all_B = np.array(all_B) / 127.5 - 1.

    return np.array(all_A), np.array(all_B)

# show 3 images per sample: original, generated, and reconstructed
# the first set is paintings, the second set photos
def save_images(ori_A, gen_B, recon_A, ori_B, gen_A, recon_B, path):

    # original painting A
    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow((ori_A * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Original")

    # painting A converted to photo B
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow((gen_B * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Generated")

    # photo B converted back to painting A
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow((recon_A * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Reconstructed")

    # original painting B
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow((ori_B * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Original")

    # painting B converted to photo A
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow((gen_A * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Generated")

    # photo A converted back to painting B
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow((recon_B * 127.5 + 127.5).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Reconstructed")

    plt.savefig(path)
    plt.close()

# test GAN and save predicted images
def evaluate_GAN(epoch, generatorAtoB, generatorBtoA):
    # get two sample sets
    batch_A, batch_B = load_test_batch()

    # generate images
    gen_B = generatorAtoB.predict(batch_A)
    gen_A = generatorBtoA.predict(batch_B)

    # Get reconstructed images
    recon_A = generatorBtoA.predict(gen_B)
    recon_B = generatorAtoB.predict(gen_A)

    # save images
    for i in range(len(gen_A)):
        where = os.path.join(OUT_DIR, "img-{}-{}".format(epoch, i))
        save_images(batch_A[i], gen_B[i], recon_A[i], batch_B[i], gen_A[i],
                    recon_B[i], path=where)


### 5. MODEL TRAINING
# train two discriminators four times
def train_discriminator(real_A, real_B):
    all_1 = np.ones(out_shape)
    all_0 = np.zeros(out_shape)

    # 1. train discriminator A using fake image A
    # 2. train discriminator B using real image B
    # returns two values: loss and accuracy
    fake_A = generatorBtoA.predict(real_B)
    A_loss_f = discriminatorA.train_on_batch(fake_A, all_0)
    A_loss_r = discriminatorA.train_on_batch(real_A, all_1)

    # 3. train discriminator B using fake image B
    # 4. train discriminator B using real image B
    # returns two values: loss and accuracy
    fake_B = generatorAtoB.predict(real_A)
    B_loss_f = discriminatorB.train_on_batch(fake_B, all_0)
    B_loss_r = discriminatorB.train_on_batch(real_B, all_1)

    # calculate the average loss
    ave = (A_loss_r[0] + A_loss_f[0] + B_loss_r[0] + B_loss_f[0]) / 4

    return ave

# train two generators at once
# generators need to fool discriminators as follows:
# 1. probabilities must be 1
# 2. reconstructed images must look real
def train_generator(batch_A, batch_B):
    all_1 = np.ones(out_shape)

    # returns 5 loss values
    # the first is the weighted some of the rest
    g_loss = cy_gan.train_on_batch([batch_A, batch_B],
                                   [all_1, all_1, batch_A, batch_B])

    return g_loss[0]

# train cycle-GAN
def train_GAN():
    print('\n=== GAN TRAINING BEGINS\n')

    # repeat epochs
    began = time()
    for epoch in range(MY_EPOCH):
        # obtain the next random batch of images
        pick = np.random.choice(num_img - MY_BATCH)
        batch_A = loadA[pick:pick + MY_BATCH]
        pick = np.random.choice(num_img - MY_BATCH)
        batch_B = loadB[pick:pick + MY_BATCH]

        # train discriminators and calculate loss
        d_loss = train_discriminator(batch_A, batch_B)

        # train generators and calculate loss
        g_loss = train_generator(batch_A, batch_B)

        if epoch % 5 == 0 or True:
            print("Epoch: {}, discriminator loss: {:.3f}, generator loss: {:.3f}"
                  .format(epoch, d_loss, g_loss))
            evaluate_GAN(epoch, generatorAtoB, generatorBtoA)

    # print training time
    total = time() - began
    print('\n=== Training Time: = {:.0f} secs, {:.1f} hrs'.format(total, total / 3600))


### 6. MODEL EVALUATION
# use cycle-GAN for prediction
def pred_GAN():
    print('\n=== GAN PREDICTION BEGINS')

    # build generator networks
    generatorAtoB = build_generator()
    generatorBtoA = build_generator()

    generatorAtoB.load_weights('./generatorAToB.h5')
    generatorBtoA.load_weights('./generatorBToA.h5')

    # get a batch of test data
    batch_A, batch_B = load_test_batch()

    # generate fake images
    gen_B = generatorAtoB.predict(batch_A)
    gen_A = generatorBtoA.predict(batch_B)

    # generate reconstructed images
    recon_A = generatorBtoA.predict(gen_B)
    recon_B = generatorAtoB.predict(gen_A)

    # save images
    for i in range(len(gen_A)):
        where = os.path.join(OUT_DIR, "pred-{}".format(i))
        save_images(batch_A[i], gen_B[i], recon_A[i], batch_B[i], gen_A[i], recon_B[i],
            path=where)


### 7. MAIN ROUTINE
# read dataset
images_A, images_B = loc_dataset()
loadA, loadB = load_images()
num_img = loadA.shape[0]

# if training is not done, we do training and save the models
# otherwise, we use the saved models to do prediction
if TRAINING:
    discriminatorA, discriminatorB = build_two_disc()
    generatorAtoB, generatorBtoA = build_two_gen()
    cy_gan = build_GAN()
    train_GAN()
    generatorAtoB.save_weights(os.path.join(OUT_DIR, 'generatorAToB.h5'))
    generatorBtoA.save_weights(os.path.join(OUT_DIR, 'generatorBToA.h5'))
else:
    pred_GAN()