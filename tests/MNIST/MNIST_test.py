import unittest
import os.path
from matplotlib import pyplot
from tests.MNIST import *
from reflect.models import SequentialModel
from reflect.layers import Relu, Dense
from reflect.optimizers import *
from reflect.regularizers import L2, L1
import numpy as np


class MNISTtest(unittest.TestCase):
    train_images    = None
    train_labels    = None
    test_images     = None
    test_labels     = None

    @classmethod
    def setUpClass(cls):
        # load train data
        cls.train_images = load_MNIST_image(TRAIN_IMAGES_URL, TRAIN_IMAGES)
        cls.train_labels = load_MNIST_label(TRAIN_LABELS_URL, TRAIN_LABELS)

        # load test data
        cls.test_images = load_MNIST_image(TEST_IMAGES_URL, TEST_IMAGES)
        cls.test_labels = load_MNIST_label(TEST_LABELS_URL, TEST_LABELS)

        cls.train_images = cls.train_images / 127 - 1.0
        cls.test_images =  cls.test_images / 127 - 1.0

        # show 9 training images

        #for i in range(9):  
        #    pyplot.subplot(330 + 1 + i)
        #    pyplot.imshow((cls.train_images[i].reshape(IMG_DIM, IMG_DIM) + 1) * 255, cmap=pyplot.get_cmap('gray'))
        #    print(cls.train_labels[i])
        #pyplot.show()

    def setUp(self):
        np.random.seed(312)

    def test_dense_gradient_descent(self):
        print("test_dense_gradient_descent()")
        steps = 3000
        step = 0.001
        hidden_1 = 300
        reg = 0.01
        examples, _ = self.train_images.shape
        test_examples, _ = self.test_images.shape

        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)

        model = SequentialModel(IMG_LENGTH)
        model.add(Dense(hidden_1, regularizer=L2(reg), 
                        weight_optimizer=GradientDescent(), bias_optimizer=GradientDescent()))
        model.add(Relu())
        model.add(Dense(10, regularizer=L2(reg), 
                        weight_optimizer=GradientDescent(), bias_optimizer=GradientDescent()))
        # output shape: (# batch, # output)
        model.compile()

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
                input.shape = (1, IMG_LENGTH)
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
            input.shape = (1, IMG_LENGTH)
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

    def test_dense_gradient_momentum(self):
        print("test_dense_gradient_momentum()")
        steps = 3000
        step = 0.001
        hidden_1 = 300
        reg = 0.01
        examples, _ = self.train_images.shape
        test_examples, _ = self.test_images.shape

        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)


        model = SequentialModel(IMG_LENGTH)
        model.add(Dense(hidden_1, regularizer=L2(reg), 
                        weight_optimizer=Momentum(), bias_optimizer=Momentum()))
        model.add(Relu())
        model.add(Dense(10, regularizer=L2(reg), 
                        weight_optimizer=Momentum(), bias_optimizer=Momentum()))
        # output shape: (# batch, # output)
        model.compile()

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
                input.shape = (1, IMG_LENGTH)
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
            input.shape = (1, IMG_LENGTH)
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
        self.assertTrue(accuracy > 50.0, "test accuracy less than 50%")

    def test_dense_gradient_RMSprop(self):
        print("test_dense_gradient_RMSprop()")
        steps = 3000
        step = 0.001
        hidden_1 = 300
        reg = 0.01
        examples, _ = self.train_images.shape
        test_examples, _ = self.test_images.shape

        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)


        model = SequentialModel(IMG_LENGTH)
        model.add(Dense(hidden_1, regularizer=L2(reg), 
                        weight_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        model.add(Relu())
        model.add(Dense(10, regularizer=L2(reg), 
                        weight_optimizer=RMSprop(), bias_optimizer=RMSprop()))
        # output shape: (# batch, # output)
        model.compile()

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
                input.shape = (1, IMG_LENGTH)
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
            input.shape = (1, IMG_LENGTH)
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
        self.assertTrue(accuracy > 50.0, "test accuracy less than 50%")
        self.assertTrue(accuracy > 50.0, "test accuracy less than 50%")
        self.assertTrue(accuracy > 50.0, "test accuracy less than 50%")

    def test_dense_adam(self):
        print("test_dense_adam()")
        steps = 3000
        step = 0.001
        hidden_1 = 300
        reg = 0.01
        examples, _ = self.train_images.shape
        test_examples, _ = self.test_images.shape

        examples = np.minimum(5000, examples)
        test_examples = np.minimum(1000, test_examples)


        model = SequentialModel(IMG_LENGTH)
        model.add(Dense(hidden_1, regularizer=L2(reg), 
                        weight_optimizer=Adam(), bias_optimizer=Adam()))
        model.add(Relu())
        model.add(Dense(10, regularizer=L2(reg), 
                        weight_optimizer=Adam(), bias_optimizer=Adam()))
        # output shape: (# batch, # output)
        model.compile()

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
                input.shape = (1, IMG_LENGTH)
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
            input.shape = (1, IMG_LENGTH)
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
        self.assertTrue(accuracy > 50.0, "test accuracy less than 50%")





if __name__ == "__main__":
    unittest.main()