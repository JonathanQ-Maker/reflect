import unittest
import os.path
from matplotlib import pyplot
from tests.MNIST import *
from reflect.models import SequentialModel
from reflect.layers import Relu, Dense, Convolve2D, Flatten, AvgPool2D, TransposedConv2D
from reflect.optimizers import *
from reflect.regularizers import L2, L1
import numpy as np

class MNISTConvolutionTest(unittest.TestCase):
    train_images    = None
    train_labels    = None
    test_images     = None
    test_labels     = None

    @classmethod
    def setUpClass(cls):
        # load train data
        cls.train_images = load_MNIST_image(TRAIN_IMAGES_URL, TRAIN_IMAGES, False)
        cls.train_labels = load_MNIST_label(TRAIN_LABELS_URL, TRAIN_LABELS)

        # load test data
        cls.test_images = load_MNIST_image(TEST_IMAGES_URL, TEST_IMAGES, False)
        cls.test_labels = load_MNIST_label(TEST_LABELS_URL, TEST_LABELS)

        cls.train_images = cls.train_images / 127 - 1.0
        cls.test_images =  cls.test_images / 127 - 1.0

        # show 9 training images

        #for i in range(9):  
        #    pyplot.subplot(330 + 1 + i)
        #    pyplot.imshow((cls.train_images[i].reshape(IMG_DIM, IMG_DIM) + 1) * 255, cmap=pyplot.get_cmap('gray'))
        #    print(cls.train_labels[i])
        #pyplot.show()
        pass

    def setUp(self):
        np.random.seed(312)

    def test_convolution(self):
        print("test_convolution()")
        steps = 3000
        step = 0.001
        kernels = 5
        filter = 3
        reg = 0.01
        examples        = self.train_images.shape[0]
        test_examples   = self.test_images.shape[0]
        
        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)

        model = SequentialModel((IMG_DIM, IMG_DIM, 1))
        model.add(Convolve2D(filter, kernels, kernel_reg=L2()))
        model.add(Relu())
        model.add(AvgPool2D(3, 2))
        model.add(Convolve2D(filter, kernels, kernel_reg=L2()))
        model.add(Relu())
        model.add(Flatten())
        model.add(Dense(10, regularizer=L2()))

        model.compile()
        model.print_summary()

        def loss(output, label):
            i = 0
            return np.log(np.sum(np.exp(output), axis=1))[i] - output[i, label]

        def dldz(output, label):
            i = 0
            d = np.exp(output) / np.sum(np.exp(output), axis=1)
            d[i, label] -= 1.0
            return d

        losses = np.zeros(examples)
        try:
            np.random.seed(312)
            for t in range(steps):
                i = np.random.randint(0, examples)
                input = self.train_images[i]
                input.shape = (1, IMG_DIM, IMG_DIM, 1)
                label = self.train_labels[i]

                model.forward(input)
                model.backprop(dldz(model.output, label))
                model.apply_grad(step)

                l = loss(model.output, label)
                losses[i] = l
                if t % 100 == 0:
                    count = np.maximum(np.count_nonzero(losses), 1)
                    print(f"[{t}]:\tloss={np.sum(losses)/count}")
        except KeyboardInterrupt:
            pass

        correct = 0
        for i in range(examples):
            input = self.train_images[i]
            input.shape = (1, IMG_DIM, IMG_DIM, 1)
            label = self.train_labels[i]

            model.forward(input)
            prediction = np.argmax(model.output, axis=1)
            if (prediction == label):
                correct += 1
        accuracy = correct * 100.0 /examples
        print(f"train accuracy: {correct}/{examples}, {round(accuracy, 2)}%")
        self.assertTrue(accuracy > 50.0, "training accuracy less than 50%")

        correct = 0
        for i in range(test_examples):
            input = self.test_images[i]
            input.shape = (1, IMG_LENGTH)
            label = self.test_labels[i]

            model.forward(input)
            prediction = np.argmax(model.output, axis=1)
            if (prediction == label):
                correct += 1
        accuracy = correct * 100.0 / test_examples
        print(f"test accuracy: {correct}/{test_examples}, {round(accuracy, 2)}%")

    def test_auto_encoder(self):
        steps = 1000
        step = 0.001
        kernels = 5
        filter = 3
        examples        = self.train_images.shape[0]
        test_examples   = self.test_images.shape[0]
        accpeted_loss   = 100
        
        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)

        model = SequentialModel((IMG_DIM, IMG_DIM, 1))

        # encoder 
        model.add(Convolve2D(filter, kernels, kernel_reg=L2()))         # 0
        model.add(Relu())                                               # 1
        model.add(AvgPool2D(3, 1))                                      # 2
        model.add(Convolve2D(filter, kernels, kernel_reg=L2()))         # 3
        model.add(Relu())                                               # 4

        # decoder
        model.add(TransposedConv2D(filter, kernels, kernel_reg=L2()))   # 5
        model.add(Relu())                                               # 6
        model.add(TransposedConv2D(3, kernels, 1, kernel_reg=L2()))     # 7
        model.add(Relu())                                               # 8
        model.add(TransposedConv2D(filter, 1, kernel_reg=L2()))         # 9

        model.compile()
        
        self.assertTrue(model.output_shape == model.input_shape, 
                        "auto encoder input-output shape diffr")

        def loss(input, output):
            residual = output - input
            return np.dot(residual.flat, residual.flat)

        def dldz(input, output):
            return output - input

        losses = np.zeros(examples)
        try:
            for t in range(steps):
                i = np.random.randint(0, examples)
                input = self.train_images[i]
                input.shape = (1, IMG_DIM, IMG_DIM, 1)
                model.forward(input)
                model.backprop(dldz(input, model.output))
                model.apply_grad(step)
                losses[i] = loss(input, model.output)
                if t % 100 == 0:
                    count = np.maximum(np.count_nonzero(losses), 1)
                    l = np.sum(losses)/count
                    print(f"[{t}]:\tloss={l}")
                    if (l < accpeted_loss):
                        break
        except KeyboardInterrupt:
            pass

        #for i in range(9):  
        #    k = i * 2
        #    output = model.forward(self.test_images[i])
        #    output = output.view()
        #    output.shape = (IMG_DIM, IMG_DIM)
        #    pyplot.subplot(3, 3*2, 1 + k)
        #    pyplot.imshow((output + 1) * 255, cmap=pyplot.get_cmap('gray'))
        #    pyplot.subplot(3, 3*2, 2 + k)
        #    pyplot.imshow((self.test_images[i] + 1) * 255, cmap=pyplot.get_cmap('gray'))
        #pyplot.show()

        count = np.maximum(np.count_nonzero(losses), 1)
        l = np.sum(losses)/count
        self.assertTrue(l < accpeted_loss, "loss higher than accpeted loss")


if __name__ == '__main__':
    unittest.main()
