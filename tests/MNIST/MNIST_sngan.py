import unittest
from tests.MNIST import *
from reflect.models import SequentialModel
from reflect.layers import Relu, LeakyRelu, Dense, DenseSN, TransposedConv2D, ConvolveSN2D, Flatten, BatchNorm, Reshape, Tanh
from reflect.optimizers import *
from reflect.regularizers import L2
from reflect.constraints import Clip
from matplotlib import pyplot
import numpy as np


class MNIST_SNgan(unittest.TestCase):
    train_images    = None
    train_labels    = None
    test_images     = None
    test_labels     = None

    @classmethod
    def setUpClass(cls):
        digit = None

        # load train data
        train_images = load_MNIST_image(TRAIN_IMAGES_URL, TRAIN_IMAGES)
        train_labels = load_MNIST_label(TRAIN_LABELS_URL, TRAIN_LABELS)

        if (digit is not None):
            cls.train_images = train_images[train_labels == digit].copy()
        cls.train_images = (train_images - 127.5) / 127.5
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
        steps = 400000
        decay_step = 200000 # step iteration when the train step would decay to min_step
        step_size = 0.0002
        min_step = 0.00001
        input_size = 100
        display_interval = 100
        n_dis = 5
        b1 = 1
        b2 = 0.1

        ###########################
        # GENERATOR
        ###########################
        generator = SequentialModel(input_size, batch_size)
        generator.add(Dense(units=7*7*128,
                                weight_optimizer=Adam(b1, b2),
                                bias_optimizer=Adam(b1, b2)))
        generator.add(Reshape(output_size=(7, 7, 128)))
        generator.add(BatchNorm(gamma_optimizer=Adam(b1, b2), bias_optimizer=Adam(b1, b2)))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(5, 5), kernels=128, 
                                       kernel_optimizer=Adam(b1, b2),
                                       bias_optimizer=Adam(b1, b2)))
        generator.add(BatchNorm(gamma_optimizer=Adam(b1, b2), bias_optimizer=Adam(b1, b2)))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(5, 5), kernels=64, strides= (2, 2),
                                       kernel_optimizer=Adam(b1, b2),
                                       bias_optimizer=Adam(b1, b2)))
        generator.add(BatchNorm(gamma_optimizer=Adam(b1, b2), bias_optimizer=Adam(b1, b2)))
        generator.add(Relu())
        generator.add(TransposedConv2D(filter_size=(4, 4), kernels=1,
                                       kernel_optimizer=Adam(b1, b2),
                                       bias_optimizer=Adam(b1, b2)))
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
        
        w_type = "he"
        discriminator = SequentialModel((28, 28, 1), 2*batch_size)
        discriminator.add(ConvolveSN2D(filter_size=(3, 3), kernels=64, strides=(1,1),
                                     weight_type=w_type,
                                     kernel_optimizer=Adam(b1, b2),
                                     bias_constraint=Clip(0)))
        discriminator.add(LeakyRelu(0.1))
        discriminator.add(ConvolveSN2D(filter_size=(4, 4), kernels=64, strides=(2,2),
                                     weight_type=w_type,
                                     kernel_optimizer=Adam(b1, b2), 
                                     bias_constraint=Clip(0)))
        discriminator.add(LeakyRelu(0.1))
        discriminator.add(ConvolveSN2D(filter_size=(3, 3), kernels=128, strides=(1,1),
                                     weight_type=w_type,
                                     kernel_optimizer=Adam(b1, b2),
                                     bias_constraint=Clip(0)))
        discriminator.add(LeakyRelu(0.1))
        discriminator.add(ConvolveSN2D(filter_size=(4, 4), kernels=128, strides=(2,2),
                                     weight_type=w_type,
                                     kernel_optimizer=Adam(b1, b2), 
                                     bias_constraint=Clip(0)))
        discriminator.add(LeakyRelu(0.1))
        discriminator.add(ConvolveSN2D(filter_size=(3, 3), kernels=256, strides=(1,1),
                                     weight_type=w_type,
                                     kernel_optimizer=Adam(b1, b2),
                                     bias_constraint=Clip(0)))
        discriminator.add(Flatten())
        discriminator.add(DenseSN(units=1,
                                weight_type=w_type,
                                weight_optimizer=Adam(b1, b2), 
                                bias_constraint=Clip(0)))
        

        generator.compile()
        discriminator.compile()
        print(f"generator params:\t{generator.total_params}")
        print(f"discriminator params:\t{discriminator.total_params}")
        
        self.assertTrue(generator.is_compiled(), "generator is not compiled")
        self.assertTrue(discriminator.is_compiled(), "discriminator is not compiled")

        dis_X = np.zeros((batch_size*2, 28, 28, 1))
        losses = []
        gen_mean = []
        dis_mean = []
        def show(save=False, name=None, loss_name=None, samples=100, seed=312):
            rng = np.random.default_rng(seed)
            size = np.ceil(np.sqrt(samples)).astype(int)
            for b in range(np.ceil(128/batch_size).astype(int)):
                X = rng.normal(size=(batch_size, input_size))
                generator.forward(X)
                np.copyto(dis_X[:batch_size], generator.output)
                np.copyto(dis_X[batch_size:], self.train_images[:batch_size])
                discriminator.forward(dis_X)
                for i in range(batch_size):
                    index = 1+i+batch_size*b
                    if (index > samples):
                        break
                    pyplot.subplot(size, size, index)
                    img = generator.output[i]
                    pyplot.imshow(img.reshape(IMG_DIM, IMG_DIM), cmap=pyplot.get_cmap('gray'))

            pyplot.setp(pyplot.gcf().get_axes(), xticks=[], yticks=[])
            if (save):
                pyplot.savefig(name)
                pyplot.clf()
                pyplot.plot(losses, label="loss")
                pyplot.plot(gen_mean, label="gen value")
                pyplot.plot(dis_mean, label="dis value")
                pyplot.legend()
                pyplot.savefig(loss_name)
            else:
                pyplot.show()
                pyplot.clf()
                pyplot.plot(losses, label="loss")
                pyplot.plot(gen_mean, label="gen value")
                pyplot.plot(dis_mean, label="dis value")
                pyplot.legend()
                pyplot.show()

        def value(real_scores, fake_scores):
            # Wasserstein Loss
            # real - fake
            return np.mean((real_scores - 1)**2) + np.mean((fake_scores + 1)**2)
        
        def dis_dl(real_score, fake_scores):
            dldz = np.zeros(discriminator.output_shape)
            dldz[:batch_size] = fake_scores + 1
            dldz[batch_size:] = real_score - 1
            return dldz
    

        def train():
            for step in range(steps):
                try:
                    if (step % display_interval == 0):
                        show(save=True, name=f"data/GAN_samples/GAN_{int(step / display_interval)}.png", 
                            loss_name=f"data/GAN_samples/GAN_loss.png")
                    
                    step_ = max(step_size - step_size * step/decay_step, min_step)
                    
                    X = np.random.randn(batch_size, input_size)
                    generator.forward(X)
                    train_batch = self.train_images[np.random.randint(0, self.TRAIN_COUNT, batch_size)]
                    np.copyto(dis_X[:batch_size], generator.output)
                    np.copyto(dis_X[batch_size:], train_batch)

                    discriminator.forward(dis_X)
                    fake_value = discriminator.output[:batch_size]
                    real_value = discriminator.output[batch_size:]
                    # calc dldz
                    dldz = dis_dl(real_value, fake_value)



                    discriminator.backprop(dldz)
                    discriminator.apply_grad(step_)
                    val = round(value(real_value, fake_value), 2)
                    losses.append(val)
                    gen_mean.append(np.mean(fake_value))
                    dis_mean.append(np.mean(real_value))

                    if (step % n_dis == 0):
                        generator.backprop(-discriminator.dldx[:batch_size])
                        generator.apply_grad(step_)
                    print(f"[{step}]:\t{val}")
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
