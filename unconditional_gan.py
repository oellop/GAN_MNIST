from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, Conv2DTranspose, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
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


(x_train, _), (_, _) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = -1)
x_train = x_train.astype('float32')/255.0

class GAN_MNIST(object):

    def __init__(self, _noise_dim = 100, dim = (28,28, 1)):
        self.generator(_noise_dim)
        self.discriminator(dim)

        self.dis.trainable = False
        self.gan = Sequential()
        self.gan.add(self.gen)
        self.gan.add(self.dis)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt, metrics= ['accuracy'])


    def generator(self, _input_shape, nodes = 128):
        self.gen = Sequential()
        self.gen.add(Dense(nodes*7*7, input_dim =_input_shape))
        self.gen.add(LeakyReLU(alpha = 0.2))
        #flat to 7*7
        self.gen.add(Reshape((7, 7, nodes)))
        #7*7 to 14*14
        self.gen.add(Conv2DTranspose(nodes,(4, 4), padding= 'same', strides = (2, 2)))
        self.gen.add(LeakyReLU(alpha = 0.2))
        #14*14 to 28*28
        self.gen.add(Conv2DTranspose(nodes,(4, 4), padding= 'same', strides = (2, 2)))
        self.gen.add(LeakyReLU(alpha = 0.2))
        self.gen.add(Conv2D(1, (7, 7), padding = 'same', activation = 'sigmoid'))



    def discriminator(self, _input_shape, nodes = 64):
        self.dis = Sequential()
        #28*28 to 14*14
        self.dis.add(Conv2D(nodes, (3, 3), input_shape = _input_shape, padding = 'same', strides = (2, 2)))
        self.dis.add(LeakyReLU(alpha = 0.2))
        self.dis.add(Dropout(0.3))
        #14*14 to 7*7
        self.dis.add(Conv2D(nodes, (3, 3), padding = 'same', strides = (2, 2)))
        self.dis.add(LeakyReLU(alpha = 0.2))
        self.dis.add(Dropout(0.3))
        self.dis.add(Flatten())
        # self.dis.add(Dropout(0.3))
        self.dis.add(Dense(1, activation = 'sigmoid'))
        self.dis.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def get_MNIST(self, num):
        return x_train[np.random.randint(0, x_train.shape[0], size=num)]

    def generate_fake_samples(self, number_sample = 256, noise_dim = 100):
        noise =  np.random.randn(number_sample*noise_dim)
        noise = noise.reshape(number_sample, noise_dim)
        gen_images = self.gen.predict(noise)
        return gen_images


    def test_accuracy(self, num=256):
        real = self.get_MNIST(num)
        real_label = np.ones((num, 1))
        _, r_acc = self.dis.evaluate(real, real_label, verbose = 0)
        fake = self.generate_fake_samples()
        fake_label = np.zeros((num, 1))
        _, f_acc = self.dis.evaluate(fake, fake_label, verbose = 0)
        print('Discriminator''s accuracy on real samples : {}'.format(r_acc))
        print('Discriminator''s accuracy on fake samples : {}'.format(f_acc))

    def train(self, epoch = 100, batch_per_epoch = 256, batch_size= 256, noise_dim = 100):
        gen_loss = []
        dis_loss = []
        half_batch = int(batch_size/2)

        for i in range(epoch):
            print('Epoch {}/{} '.format(i, epoch))
            for j in (range(batch_per_epoch)):
                real = self.get_MNIST(half_batch)
                fake = self.generate_fake_samples(number_sample = half_batch, noise_dim = noise_dim)

                images = np.concatenate([real, fake])
                labels = np.ones(images.shape[0])
                labels[half_batch:] = 0
                d_loss,_ = self.dis.train_on_batch(images, labels)

                noise =  np.random.randn(half_batch*noise_dim)
                noise = noise.reshape(half_batch, noise_dim)
                label = np.ones((half_batch,1))

                g_loss,_ = self.gan.train_on_batch(noise, label)
                print('{}/{}  d_loss : {}  g_loss : {}'.format(j, batch_per_epoch, d_loss, g_loss))

            self.test_accuracy()
            self.gen.save('generator_model.h5')
            self.dis.save('discriminator_model.h5')
            if i%5==0:
                self.plot(i)


    def test_loaded_model(self):
        self.gen = load_model('generator_model.h5')
        images = self.generate_fake_samples()
        number = 10
        for i in range(number*number):
            plt.subplot(number, number, i+1)
            plt.imshow(images[i, :, :, 0], cmap = 'gray_r')
            plt.axis('off')
        plt.savefig('generated_handwritten_digit')
        plt.clf()


    def plot(self,epoch, number=10):
        images = self.generate_fake_samples(number_sample = number*number)
        # images = self.get_MNIST(number*number)
        images = images.reshape(number*number, 28, 28)

        for i in range(number*number):
            plt.subplot(number, number, i+1)
            plt.imshow(images[i], cmap = 'gray_r')
            plt.axis('off')

        plt.savefig('uGan_epoch/gan_epoch_{}'.format(epoch))
        plt.clf()

a = GAN_MNIST()
a.test_loaded_model()
# a.train()
