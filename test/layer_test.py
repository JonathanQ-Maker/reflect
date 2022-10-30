from reflect.layers import Dense
import numpy as np
from reflect.regularizers import L1, L2, L1L2
from reflect.profiler.numerical_gradient import num_grad

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
    l.compile(gen_param=True)

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
        l.backprop(residual)
        l.apply_grad(step)

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
        l.compile(gen_param=True)
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
            dldx = layers[i].backprop(dldx)
            layers[i].apply_grad(step)

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
    assert not l.is_compiled(), "Dense layer should not be compiled"
    assert not reg.is_compiled(), "Regularizer should not be compiled"

    np.random.seed(0)
    l.compile(gen_param=True)
    np.random.seed(0)
    ctrl.compile(gen_param=True)

    assert l.param is not ctrl.param, "layer and control layer param is the same"
    assert np.all(l.param.weight == ctrl.param.weight), "Layers weights are different"

    assert l.is_compiled(), "Dense layer is not compiled"
    assert reg.is_compiled(), "Regularizer is not compiled"
    assert reg.grad.shape == l.weight_shape, "Regularizer grad.shape != weight.shape"

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
            d.backprop(residual)
            d.apply_grad(step)

            if (i % 10 == 0):
                print(f"Loss[{i}]: {loss}")
        final_loss = loss
        print(f"Final loss - Initial Loss = {final_loss - init_loss}")
        assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"

    l_weight_norm = np.linalg.norm(l.param.weight)
    ctrl_weight_norm = np.linalg.norm(ctrl.param.weight)
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
    assert not l.is_compiled(), "Dense layer should not be compiled"
    assert not reg.is_compiled(), "Regularizer should not be compiled"

    np.random.seed(0)
    l.compile(gen_param=True)
    np.random.seed(0)
    ctrl.compile(gen_param=True)

    assert l.param is not ctrl.param, "layer and control layer param is the same"
    assert np.all(l.param.weight == ctrl.param.weight), "Layers weights are different"

    assert l.is_compiled(), "Dense layer is not compiled"
    assert reg.is_compiled(), "Regularizer is not compiled"
    assert reg.grad.shape == l.weight_shape, "Regularizer grad.shape != weight.shape"

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
            d.backprop(residual)
            d.apply_grad(step)

            if (i % 10 == 0):
                print(f"Loss[{i}]: {loss}")
        final_loss = loss
        print(f"Final loss - Initial Loss = {final_loss - init_loss}")
        assert np.all(final_loss < init_loss), "final loss is not smaller than initial loss"

    l_weight_norm = np.linalg.norm(l.param.weight)
    ctrl_weight_norm = np.linalg.norm(ctrl.param.weight)
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

def dense_param_switch_test():
    input_size = 5
    output_size = 10
    batch_size = 2
    count = 100
    step = 0.01

    l = Dense(input_size, output_size, batch_size, "xavier", L2(0.01))
    l.compile(gen_param=True)
    param = l.param
    param2 = l.create_param()

    assert param is not param2, "params the same instance"
    assert np.all(param.bias.shape == param2.bias.shape), "bias does not match"
    assert np.all(param.weight.shape == param2.weight.shape), "weights does not match"
    assert param.weight_type == param2.weight_type, "weight types does not match"
    assert param.regularizer is not param2.regularizer, "regularizer is the same insatnce"

    assert l.param_compatible(param), "generated param is not compatible"

    np.copyto(param2.weight, param.weight)

    input = np.random.uniform(size=l.input_shape)
    target = np.random.uniform(size=l.output_shape)

    inital_output = np.copy(l.forward(input))

    for i in range(count):
        output = l.forward(input)
        residual = target - output
        loss = np.sum(residual ** 2)
        if (i % 10 == 0):
            print(f"loss[{i}]: {np.round(loss, 3)}")
        l.backprop(residual)
        l.apply_grad(step)

    param_output = np.copy(l.forward(input))
    l.apply_param(param2)
    param2_output = np.copy(l.forward(input))

    assert np.all(param_output != param2_output), "some param outputs are same"
    assert np.any(inital_output != param_output), "layer output is still the same after backprop"
    assert np.all(inital_output == param2_output), "layer output is not the same when switched to original param"
    print("dense_param_switch_test() passed")

def dense_num_grad_test():
    np.set_printoptions(precision=8)
    input_size = 5
    output_size = 3
    batch_size = 2
    count = 100
    step = 0.01


    l = Dense(input_size, output_size, batch_size, "xavier")
    l.compile(gen_param=True)

    input = np.random.uniform(size=l.input_shape)
    target = np.random.uniform(size=l.output_shape)
    weight_original = l.param.weight
    weight = np.copy(l.param.weight)

    def forward(W):
        l.param.weight = W
        return np.sum((target - l.forward(input))**2) / 2

    grad = num_grad(forward, weight)

    l.param.weight = weight_original
    residual = target - l.forward(input)
    l.backprop(residual)
    real_grad = -l.dldw

    assert np.all(grad != real_grad), "grad == real_grad strictly"
    assert grad is not real_grad, "grad and real_grad is the same instance"
    print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
    assert np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ"
    print("dense_num_grad_test() passed\n")

def multi_dense_num_grad_test():
    np.set_printoptions(precision=8)
    input_size = 5
    output_size_1 = 7
    output_size_2 = 3
    batch_size = 2


    l1 = Dense(input_size, output_size_1, batch_size, "xavier")
    l2 = Dense(output_size_1, output_size_2, batch_size, "xavier")
    l1.compile(gen_param=True)
    l2.compile(gen_param=True)

    input = np.random.uniform(size=l1.input_shape)
    target = np.random.uniform(size=l2.output_shape)
    weight_original = l1.param.weight
    weight = np.copy(l1.param.weight)

    def forward(W):
        l1.param.weight = W
        return np.sum((target - l2.forward(l1.forward(input)))**2) / 2

    grad = num_grad(forward, weight)

    l1.param.weight = weight_original
    residual = target - l2.forward(l1.forward(input))
    l1.backprop(l2.backprop(residual))
    real_grad = -l1.dldw

    assert np.all(grad != real_grad), "grad == real_grad strictly"
    assert grad is not real_grad, "grad and real_grad is the same instance"
    print(f"grad:\n{grad}\n\nreal_grad:\n{real_grad}")
    assert np.allclose(grad, real_grad, atol = 1e-4), "num_grad and real gradient differ"
    print("multi_dense_num_grad_test() passed\n")




    





if (__name__ == "__main__"):
    dense_one_layer_test()
    dense_multi_layer_test()
    dense_one_layer_l1_regularizer_test()
    dense_one_layer_l2_regularizer_test()
    regularizers_test()
    dense_param_switch_test()
    dense_num_grad_test()
    multi_dense_num_grad_test()
