from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


class DCGAN(object):
    """
    Mnist手写数字图片的生成
    """
    def __init__(self):
        # 输入图片的形状
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100  # 生成原始噪点数据，噪点数据向量长度大小

        optimizer = Adam(0.0002, 0.5)

        # 建立判别器训练模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # 联合建立生成器训练参数，指定生成器损失
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        # 输入噪点向量，使用生成器生成图片
        z = Input(shape=(self.latent_dim, ))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)  # 这里是训练好的判别器

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        noise = Input(shape=(self.latent_dim, ))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # def load_data_DCGAN(self, path="MNIST_data/mnist.npz"):
    #     f = np.load(path)
    #     x_train, y_train = f['x_train'], f['y_train']
    #     x_test, y_test = f['x_test'], f['y_test']
    #     f.close()
    #     return (x_train, y_train), (x_test, y_test)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset  X_train.shape = (60000, 28, 28)
        (X_train, _), (_, _) = mnist.load_data_DCGAN()  # 这里由于下载数据集不成功，所以自己修改了load_data()

        # Rescale -1 to 1  归一化处理
        X_train = X_train / 127.5 - 1.
        # 扩充X_train的形状，因为神经网络的输入要求数据是4维  X_train.shape = (60000, 28, 28) ---> (60000, 28, 28, 1)
        X_train = np.expand_dims(X_train, axis=3)

        # 正负样本的目标值建立
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # --------------------
            # Train Discriminator
            # --------------------

            # 随机选取一些真实样本
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 生成器生成假样本
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)  # imgs为真实样本的特征值，valid为真实样本的目标值
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # 计算平均值，d_loss = (损失，准确率)

            # --------------------
            # Train Generator
            # --------------------

            # Train the generator (wants discriminator to mistake images as real)
            # 训练生成器，停止训练判别器
            # 用目标值为1去训练，目的使得生成器生成的样本越来越接近真实样本
            g_loss = self.combined.train_on_batch(noise, valid)  # valid全是1，即使用噪声去训练模型，使得它越来越接近于1

            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r *c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('images/mnist_%d.png' % epoch)
        plt.close()


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)