from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(object):
    """
    实现自动编码器
    定义编码器：输出32个神经元，使用relu激活函数
    定义解码器：输出784个神经元，使用sigmoid函数 （784这个值是输出与原来图片大小一致）
    """
    def __init__(self):
        self.encoding_dim = 32  # 编码器输出神经元个数
        self.decoding_dim = 784  # 解码器输出神经元个数
        self.model = self.auto_encoder_model_3()

    def auto_encoder_model_1(self):
        """
        初始化普通自动编码器模型
        将编码器和解码器放在一起作为一个模型
        :return: auto_encoder
        """
        input_img = Input(shape=(784, ))  # 定义输入形状
        encoder = Dense(self.encoding_dim, activation='relu')(input_img)
        decoder = Dense(self.decoding_dim, activation='sigmoid')(encoder)

        # 定义完整的模型结构，输入图片，输出decoder的结果
        auto_encoder = Model(inputs=input_img, outputs=decoder)
        auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        return auto_encoder

    def auto_encoder_model_2(self):
        """
        实现复杂编码器模型
        其实就是多个简单编码器进行累加在一起
        """
        input_img = Input(shape=(784, ))
        encoder = Dense(128, activation='relu')(input_img)
        encoder = Dense(64, activation='relu')(encoder)
        encoder = Dense(32, activation="relu")(encoder)

        decoder = Dense(64, activation='relu')(encoder)
        decoder = Dense(128, activation='relu')(decoder)
        decoder = Dense(784, activation='sigmoid')(decoder)

        auto_encoder = Model(input=input_img, output=decoder)
        auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        return auto_encoder

    def auto_encoder_model_3(self):
        """
        实现卷积编码器模型
        """
        input_img = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoder = MaxPooling2D((2, 2), padding='same')(x)
        # print(encoder)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 这里filter个数为1，因为图片的通道数为1
        # print(decoder)

        auto_encoder = Model(input_img, decoder)
        auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        return auto_encoder


    def train(self):
        """
        训练流程
            读取mnist数据，并进行归一化处理以及形状修改，使得图片数据可以输入到模型中
            模型进行fit训练
                指定迭代次数
                指定每批次数据大小
                是否打乱数据
                验证集合
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()

        # 进行归一化
        x_train = x_train.astype('float32') / 255.  # x_train.shape =  (60000, 28, 28)
        x_test = x_test.astype('float32') / 255.  # x_test.shape =  (10000, 28, 28)
        # print("x_train.shape = ", x_train.shape)
        # print("x_test.shape = ", x_test.shape)

        # 改变形状
        # np.prod(x_train.shape[1:])
        # x_train.shape有3维，上面的代码是让第2个维度与第3个维度进行相乘，即28 * 28 = 784
        x_train = np.reshape(x_train, (len(x_train), np.prod(x_train.shape[1:])))  # x_train.shape = (60000, 784)
        x_test = np.reshape(x_test, (len(x_test), np.prod(x_test.shape[1:])))  # x_test.shape =  (10000, 784)
        # print("x_train.shape = ", x_train.shape)
        # print("x_test.shape = ", x_test.shape)

        # 特征是为x_train，目标值也是x_train，因为自己编码自己解码，将生成的图片与原来的图片进行对比
        self.model.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    def train_conv(self):
        """
        适用于卷积编码器的训练流程
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()
        x_train = x_train[0: 10000, :, :]
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        # print(x_train.shape)
        # print(x_test.shape)

        self.model.fit(x_train, x_train, epochs=5, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    def train_noise(self):
        """
        降噪编码器
        只是在上面两种卷积编码器的训练过程中加入噪点
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()
        x_train = x_train[0: 10000, :, :]
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        # 加入噪点
        x_train_noise = x_train + np.random.normal(loc=0.0, scale=3.0, size=x_train.shape)
        x_test_noise = x_test + np.random.normal(loc=0.0, scale=3.0, size=x_test.shape)

        # 重新进行限制每个像素值的大小在0~1之间
        x_train_noise = np.clip(x_train_noise, 0., 1.)
        x_test_noise = np.clip(x_test_noise, 0., 1.)

        self.model.fit(x_train_noise, x_train_noise, epochs=5, batch_size=256, shuffle=True, validation_data=(x_test_noise, x_test_noise))

    def display(self):
        """
        显示前后效果对比
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()
        x_test = np.reshape(x_test, (len(x_test), np.prod(x_test.shape[1:])))

        decoded_img = self.model.predict(x_test)

        plt.figure(figsize=(20, 4))

        # 显示5张图片
        n = 5
        for i in range(n):
            # 显示编码前结果
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 显示编码后结果
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_img[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def display_conv(self):
        """
        使用卷积网络进行训练后，显示前后效果对比
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()
        x_test = np.expand_dims(x_test, axis=3)

        decoded_img = self.model.predict(x_test)

        plt.figure(figsize=(20, 4))

        # 显示5张图片
        n = 5
        for i in range(n):
            # 显示编码前结果
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 显示编码后结果
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_img[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def display_noise(self):
        """
        加入噪点后，显示前后效果对比
        """
        (x_train, _), (x_test, _) = mnist.load_data_DCGAN()
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
        x_test_noise = x_test + np.random.normal(loc=3.0, scale=10.0, size=x_test.shape)

        decoded_img = self.model.predict(x_test_noise)

        plt.figure(figsize=(20, 4))

        # 显示5张图片
        n = 5
        for i in range(n):
            # 显示编码前结果
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test_noise[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 显示编码后结果
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_img[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == "__main__":
    autodecoder = AutoEncoder()
    autodecoder.train_noise()
    autodecoder.display_noise()
