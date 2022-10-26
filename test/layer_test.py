from reflect.layers import Dense
import numpy as np
from reflect.regularizers import L1, L2, L1L2

np.set_printoptions(precision=3)
np.random.seed(0)

def dense_one_layer_test():
    batch_size = 2
    input_size = 5
    output_size = 3
    step = 0.001
    step_count = 100

    input_shape = (batch_size, input_size)
    output_shape = (batch_size, output_size)

    l = Dense(input_size, output_size, batch_size, "xavier")
    assert not l.is_compiled(), "Dense layer should not be compiled"
    l.compile()

    assert l.is_compiled(), "Dense layer is not compiled"
    print(str(l))

    input = np.random.uniform(size=input_shape)
    target = np.random.uniform(size=output_shape)
    
    output = l.forward(input)
    residual = target - output
    loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
    init_loss = loss
    for i in range(step_count):
        output = l.forward(input)
        residual = target - output
        loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
        l.backprop(residual, step)

        if (i % 10 == 0):
            print(f"Loss[{i}]: {loss}")
    final_loss = loss
    print(f"Final loss - Initial Loss = {final_loss - init_loss}")
    assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"
    print("dense_one_layer_test() passed\n")

def dense_multi_layer_test():
    batch_size = 20
    dimentions = [50, 40, 30, 70, 60, 50, 40, 30]
    input_shape = (batch_size, dimentions[0])
    output_shape = (batch_size, dimentions[-1])
    step = 0.0001
    step_count = 100

    input = np.random.uniform(size=input_shape)
    target = np.random.uniform(size=output_shape)
    
    layers = []
    for i in range(len(dimentions) - 1):
        l = Dense(dimentions[i], dimentions[i + 1], batch_size, "xavier")
        assert not l.is_compiled(), "Dense layer should not be compiled"
        l.compile()
        assert l.is_compiled(), f"Layer {i} is not compiled"
        layers.append(l)

    init_loss = 0
    final_loss = 0
    for e in range(step_count):
        output = input
        for i in range(len(layers)):
            output = layers[i].forward(output)
            assert np.std(output) > 0.1, "std at layer {i} is less than 0.1"
            if (e == 0):
                print(f"layer[{i}] out std: {np.std(output)}")

        residual = target - output
        loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
        if (e == 0):
            init_loss = loss
        elif (e == step_count - 1):
            final_loss = loss

        if (e % 10 == 0):
            print(f"loss[{e}]: {loss}")

        dldx = residual
        for i in reversed(range(len(layers))):
            dldx = layers[i].backprop(dldx, step)

    print(f"Final loss - Initial Loss = {final_loss - init_loss}")
    assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"
    print("dense_multi_layer_test() passed\n")

def dense_one_layer_l1_regularizer_test():
    batch_size = 2
    input_size = 5
    output_size = 3
    step = 0.001
    step_count = 100

    input_shape = (batch_size, input_size)
    output_shape = (batch_size, output_size)

    coeff = 0.5

    reg = L1(coeff)
    l = Dense(input_size, output_size, batch_size, "xavier", reg)
    ctrl = Dense(input_size, output_size, batch_size, "xavier")
    assert np.all(l.weight == ctrl.weight), "Layers weights are different"
    assert not l.is_compiled(), "Dense layer should not be compiled"
    assert not reg.is_compiled(), "Regularizer should not be compiled"

    np.random.seed(0)
    l.compile()
    np.random.seed(0)
    ctrl.compile()

    assert l.is_compiled(), "Dense layer is not compiled"
    assert reg.is_compiled(), "Regularizer is not compiled"
    assert reg.grad.shape == l.weight.shape, "Regularizer grad.shape != weight.shape"

    input = np.random.uniform(size=input_shape)
    target = np.random.uniform(size=output_shape)
    
    for d in [l, ctrl]:
        output = d.forward(input)
        residual = target - output
        loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
        init_loss = loss
        for i in range(step_count):
            output = d.forward(input)
            residual = target - output
            loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
            d.backprop(residual, step)

            if (i % 10 == 0):
                print(f"Loss[{i}]: {loss}")
        final_loss = loss
        print(f"Final loss - Initial Loss = {final_loss - init_loss}")
        assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"

    l_weight_norm = np.linalg.norm(l.weight)
    ctrl_weight_norm = np.linalg.norm(ctrl.weight)
    assert l_weight_norm < ctrl_weight_norm, "regularized weight norm bigger than non regularized"
    print(f"l_weight_norm: {np.round(l_weight_norm, 3)}, ctrl_weight_norm: {np.round(ctrl_weight_norm, 3)}")

    print("dense_one_layer_l1_regularizer_test() passed\n")

def dense_one_layer_l2_regularizer_test():
    batch_size = 2
    input_size = 5
    output_size = 3
    step = 0.001
    step_count = 100

    input_shape = (batch_size, input_size)
    output_shape = (batch_size, output_size)

    coeff = 0.5

    reg = L2(coeff)
    l = Dense(input_size, output_size, batch_size, "xavier", reg)
    ctrl = Dense(input_size, output_size, batch_size, "xavier")
    assert np.all(l.weight == ctrl.weight), "Layers weights are different"
    assert not l.is_compiled(), "Dense layer should not be compiled"
    assert not reg.is_compiled(), "Regularizer should not be compiled"

    np.random.seed(0)
    l.compile()
    np.random.seed(0)
    ctrl.compile()

    assert l.is_compiled(), "Dense layer is not compiled"
    assert reg.is_compiled(), "Regularizer is not compiled"
    assert reg.grad.shape == l.weight.shape, "Regularizer grad.shape != weight.shape"

    input = np.random.uniform(size=input_shape)
    target = np.random.uniform(size=output_shape)
    
    for d in [l, ctrl]:
        output = d.forward(input)
        residual = target - output
        loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
        init_loss = loss
        for i in range(step_count):
            output = d.forward(input)
            residual = target - output
            loss = np.sum(residual ** 2) / (batch_size * output_shape[1])
            d.backprop(residual, step)

            if (i % 10 == 0):
                print(f"Loss[{i}]: {loss}")
        final_loss = loss
        print(f"Final loss - Initial Loss = {final_loss - init_loss}")
        assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"

    l_weight_norm = np.linalg.norm(l.weight)
    ctrl_weight_norm = np.linalg.norm(ctrl.weight)
    assert l_weight_norm < ctrl_weight_norm, "regularized weight norm bigger than non regularized"
    print(f"l_weight_norm: {np.round(l_weight_norm, 3)}, ctrl_weight_norm: {np.round(ctrl_weight_norm, 3)}")

    print("dense_one_layer_l2_regularizer_test() passed\n")

def regularizers_test():
    coeff = 0.01
    regl1 = L1(coeff)
    regl2 = L2(coeff)
    regl1l2 = L1L2(coeff, coeff)

    shape = (5, 3)

    weight = np.random.normal(size=shape)
    regl1.shape = shape
    regl2.shape = shape
    regl1l2.shape = shape

    l1_grad = regl1.gradient(weight)
    l2_grad = regl2.gradient(weight)
    l1l2_grad = regl1l2.gradient(weight)

    l1_grad_norm = np.linalg.norm(l1_grad)
    l2_grad_norm = np.linalg.norm(l2_grad)
    l1l2_grad_norm = np.linalg.norm(l1l2_grad)

    assert l1_grad_norm < l1l2_grad_norm and l2_grad_norm < l1l2_grad_norm, "l1l2 gradient norm smaller"
    print("regularizers_test() passed")





if (__name__ == "__main__"):
    dense_one_layer_test()
    dense_multi_layer_test()
    dense_one_layer_l1_regularizer_test()
    dense_one_layer_l2_regularizer_test()
    regularizers_test()
