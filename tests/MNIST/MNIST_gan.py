import unittest
from tests.MNIST import *
from reflect.models import SequentialModel
from reflect.layers import Relu, Dense, TransposedConv2D, Convolve2D, Flatten, BatchNorm, Reshape, Tanh
from reflect.optimizers import *
from reflect.regularizers import L2
from reflect.constraints import Clip
from matplotlib import pyplot
import numpy as np


class MNISTGan(unittest.TestCase):
    train_images    = None
    train_labels    = None
    test_images     = None
    test_labels     = None

    @classmethod
    def setUpClass(cls):
        digit = 2

        # load train data
        train_images = load_MNIST_image(TRAIN_IMAGES_URL, TRAIN_IMAGES)
        train_labels = load_MNIST_label(TRAIN_LABELS_URL, TRAIN_LABELS)

        if (digit is not None):
            cls.train_images = train_images[train_labels == digit].copy()
        cls.train_images = (cls.train_images - 127.5) / 127.5
        cls.TRAIN_COUNT, _ = cls.train_images.shape
        cls.train_images.shape = (cls.TRAIN_COUNT, IMG_DIM, IMG_DIM, 1)

        print(f"train count: {cls.TRAIN_COUNT}")

        # show 9 training images

        #for i in range(9):  
        #    pyplot.subplot(330 + 1 + i)
        #    pyplot.imshow(cls.train_images[i].reshape(IMG_DIM, IMG_DIM), cmap=pyplot.get_cmap('gray'))
        #pyplot.show()


    def setUp(self):
        np.random.seed(312)

    def test(self):
        batch_size = 16
        epochs = 300
        discriminator_step = 0.00005
        generator_step = 0.00005
        input_size = 100

        ###########################
        # GENERATOR
        ###########################
        generator = SequentialModel(input_size, batch_size)
        generator.add(Dense(units=7*7*128, weight_reg=L2(),
                                weight_optimizer=RMSprop(), 
                                bias_constraint=Clip(0)))
        generator.add(Reshape(output_size=(7, 7, 128)))
        generator.add(BatchNorm(gamma_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(5, 5), kernels=128, 
                                       kernel_reg=L2(),
                                       kernel_optimizer=RMSprop(), 
                                       bias_constraint=Clip(0)))
        generator.add(BatchNorm(gamma_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(5, 5), kernels=64, strides= (2, 2),
                                       kernel_reg=L2(),
                                       kernel_optimizer=RMSprop(), 
                                       bias_constraint=Clip(0)))
        generator.add(BatchNorm(gamma_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(4, 4), kernels=1,
                                       kernel_reg=L2(),
                                       kernel_optimizer=RMSprop(), 
                                       bias_constraint=Clip(0)))
        generator.add(Tanh())
        ###########################
        # DISCRIMINATOR
        ###########################

        # INPUT FORMAT
        # assuming batch_size = 16
        #    *------------*
        # 0  | fake input |
        # 1  | fake input |
        # .  | ... 16x    |
        # .  |------------|
        # 16 | real input |
        # 17 | real input |
        # .  | ... 16x    |
        #    *------------*

        clip = 0.01
        discriminator = SequentialModel((28, 28, 1), batch_size * 2)
        discriminator.add(Convolve2D(filter_size=(4, 4), kernels=64, strides=(2,2),
                                     kernel_reg=L2(),
                                     kernel_optimizer=RMSprop(), 
                                     kernel_constraint=Clip(clip),
                                     bias_constraint=Clip(0)))
        discriminator.add(BatchNorm(gamma_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        discriminator.add(Relu())
        discriminator.add(Convolve2D(filter_size=(3, 3), kernels=128, strides=(2,2),
                                     kernel_reg=L2(),
                                     kernel_optimizer=RMSprop(), 
                                     kernel_constraint=Clip(clip),
                                     bias_constraint=Clip(0)))
        discriminator.add(BatchNorm(gamma_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        discriminator.add(Relu())
        discriminator.add(Flatten())
        discriminator.add(Dense(units=1, weight_reg=L2(),
                                weight_optimizer=RMSprop(), 
                                weight_constraint=Clip(clip),
                                bias_constraint=Clip(0)))

        generator.compile()
        discriminator.compile()
        print(f"generator params:\t{generator.total_params}")
        print(f"discriminator params:\t{discriminator.total_params}")
        
        self.assertTrue(generator.is_compiled(), "generator is not compiled")
        self.assertTrue(discriminator.is_compiled(), "discriminator is not compiled")

        losses = []
        dldz = np.repeat([[-1], [1]], (batch_size, batch_size), axis=0)
        discriminator_X = np.random.randn(batch_size*2, 28, 28, 1)
        seed = np.random.randn(batch_size, input_size)
        def show(save=False, name=None, loss_name=None):
            generator.forward(seed)
            np.copyto(discriminator_X[:batch_size], generator.output)
            discriminator.forward(discriminator_X)
            for i in range(np.minimum(16,batch_size)):  
                pyplot.subplot(4, 4, 1+i)
                img = generator.output[i]
                pyplot.imshow(img.reshape(IMG_DIM, IMG_DIM), cmap=pyplot.get_cmap('gray'))
                print(discriminator.output[i, 0])
            pyplot.setp(pyplot.gcf().get_axes(), xticks=[], yticks=[])
            if (save):
                pyplot.savefig(name)
                pyplot.clf()
                pyplot.plot(losses)
                pyplot.savefig(loss_name)
            else:
                pyplot.show()
                pyplot.clf()
                pyplot.plot(losses)
                pyplot.show()

        def value(scores):
            # Wasserstein Loss
            # real - fake
            return np.mean(scores[batch_size:] - scores[:batch_size])

        def train():
            batch_count = int(self.TRAIN_COUNT / batch_size)
            for epoch in range(epochs):
                show(save=True, name=f"data/GAN_samples/GAN_{epoch}.png", 
                         loss_name=f"data/GAN_samples/GAN_loss.png")
                for batch in range(batch_count):
                    try:
                        X = np.random.randn(batch_size, input_size)
                        generator.forward(X)
                        np.copyto(discriminator_X[:batch_size], generator.output)
                        np.copyto(discriminator_X[batch_size:], self.train_images[batch*batch_size:
                                                                                  (batch+1)*batch_size])
                        discriminator.forward(discriminator_X)
                        discriminator.backprop(dldz)
                        val = round(value(discriminator.output), 2)
                        losses.append(val)

                        if ((batch_count*epoch + batch + 1) % 5 == 0):
                            generator.backprop(-discriminator.dldx[:batch_size])
                            generator.apply_grad(generator_step)
                        else:
                            discriminator.apply_grad(discriminator_step)
                        print(f"[{batch_count*epoch + batch + 1}]:\t{val}")
                    except KeyboardInterrupt:
                        print(generator)
                        print(discriminator)
                        show()

                        if (input("continue? (y/n)") == "n"):
                            return

        train()
        show()

                


if __name__ == "__main__":
    unittest.main()
