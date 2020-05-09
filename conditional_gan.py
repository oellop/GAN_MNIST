from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, Conv2DTranspose, Reshape, Embedding, Concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from keras.utils.vis_utils import plot_model
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle


opt = Adam(lr=0.0002, beta_1=0.5)

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)


(x_train, y_train), (_, _) = mnist.load_data()

x_train = np.expand_dims(x_train, axis = -1)
x_train = x_train.astype('float32')/255.0

class cGAN_MNIST(object):

    def __init__(self, _noise_dim = 100, dim = (28,28, 1)):
        self.generator(_noise_dim)
        self.discriminator(dim)
        self.dis.trainable = False
        gen_label, gen_noise = self.gen.input
        gen_output = self.gen.output
        gan_output = self.dis([gen_label, gen_output])
        self.gan = Model([gen_label, gen_noise], gan_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt, metrics= ['accuracy'])


    def generator(self, _input_shape, nodes = 128):
        in_label = Input(shape=(1,), name= 'Label_input')

        emb = Embedding(10, 50)(in_label)
        emb = Dense(7*7)(emb)
        emb = Reshape((7,7,1))(emb)

        in_noise = Input(shape=(_input_shape,), name = 'Noise_input')
        gen = Dense(nodes*7*7)(in_noise)
        gen = LeakyReLU(alpha = 0.2)(gen)
        gen = Reshape((7,7, nodes))(gen)
        gen = Concatenate()([emb, gen])
        #7*7 to 14*14
        gen = Conv2DTranspose(nodes,(4, 4), padding= 'same', strides = (2, 2))(gen)
        gen = LeakyReLU(alpha = 0.2)(gen)
        #14*14 to 28*28
        gen = Conv2DTranspose(nodes,(4, 4), padding= 'same', strides = (2, 2))(gen)
        gen = LeakyReLU(alpha = 0.2)(gen)
        out_layer = Conv2D(1,(7, 7), padding= 'same', activation = 'tanh')(gen)

        self.gen = Model([in_label, in_noise], out_layer)

    def discriminator(self, _input_shape, nodes = 64):
        in_label = Input(shape = (1,), name='Label_input')

        emb = Embedding(10, 50)(in_label)
        emb = Dense(28*28)(emb)
        emb = Reshape((28,28,1))(emb)

        in_img = Input(shape=(_input_shape), name='Image_input')
        disc = Concatenate()([emb, in_img])
        disc = Conv2D(nodes, (3, 3), padding = 'same', strides = (2, 2))(disc)
        disc = LeakyReLU(alpha = 0.2)(disc)
        disc = Dropout(0.3)(disc)
        disc = Conv2D(nodes, (3, 3), padding = 'same', strides = (2, 2))(disc)
        disc = LeakyReLU(alpha = 0.2)(disc)
        disc = Dropout(0.3)(disc)
        disc = Flatten()(disc)
        out_layer = Dense(1, activation = 'sigmoid')(disc)
        self.dis = Model([in_label, in_img], out_layer)
        self.dis.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


    def get_MNIST(self, num):
        rand_indice = np.random.randint(0, x_train.shape[0], size=num)
        return x_train[rand_indice], y_train[rand_indice]

    def generate_fake_samples(self, number_sample = 256, noise_dim = 100):
        noise =  np.random.randn(number_sample*noise_dim)
        noise = noise.reshape(number_sample, noise_dim)
        y = np.random.randint(0, 10, size = number_sample)
        gen_images = self.gen.predict([y, noise])
        return gen_images, y


    def train(self, epoch = 100, batch_per_epoch = 256, batch_size= 256, noise_dim = 100):
        gen_loss = []
        dis_loss = []
        half_batch = int(batch_size/2)

        for i in range(epoch):
            print('Epoch {}/{} '.format(i, epoch))
            for j in (range(batch_per_epoch)):
                x_real, y_real = self.get_MNIST(half_batch)
                x_fake, y_fake = self.generate_fake_samples(number_sample = half_batch, noise_dim = noise_dim)

                images = np.concatenate([x_real, x_fake])
                y = np.concatenate([y_real, y_fake])
                labels = np.ones(images.shape[0])
                labels[half_batch:] = 0
                d_loss,_ = self.dis.train_on_batch([y, images], labels)

                noise =  np.random.randn(half_batch*noise_dim)
                noise = noise.reshape(half_batch, noise_dim)
                y = np.random.randint(0, 10, half_batch)

                label = np.ones((half_batch,1))

                g_loss,_ = self.gan.train_on_batch([y,noise], label)
                print('{}/{}  d_loss : {}  g_loss : {}'.format(j, batch_per_epoch, d_loss, g_loss))

            # self.test_accuracy()
            self.gen.save('generator_cmodel.h5')
            self.dis.save('discriminator_cmodel.h5')
            if i%5==0:
                self.plot(i)


    def test_loaded_model(self, number=10):
        self.gen = load_model('generator_cmodel.h5')
        labels = [i%10 for i in range(number*number)]
        labels = np.expand_dims(labels, axis = -1)
        noise =  np.random.randn(number*number*100)
        noise = noise.reshape(number*number, 100)
        images = self.gen.predict([labels, noise])
        for i in range(number*number):
            plt.subplot(number, number, i+1)
            plt.imshow(images[i, :, :, 0], cmap = 'gray_r')
            plt.axis('off')

        plt.savefig('generated_handwritten_digit_cGan')
        plt.clf()


    def plot(self,epoch, number=10):
        labels = [i%10 for i in range(number*number)]
        labels = np.expand_dims(labels, axis = -1)
        noise =  np.random.randn(number*number*100)
        noise = noise.reshape(number*number, 100)
        images = self.gen.predict([labels, noise])
        for i in range(number*number):
            plt.subplot(number, number, i+1)
            plt.imshow(images[i, :, :, 0], cmap = 'gray_r')
            plt.axis('off')

        plt.savefig('cGan_epoch/cGan_epoch_{}'.format(epoch))
        plt.clf()


a = cGAN_MNIST()
a.test_loaded_model()
# a.train()
