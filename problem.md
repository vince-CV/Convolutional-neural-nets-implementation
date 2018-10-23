## Ground rules

You may use your programming language of choice. You may only use your language's standard library, with the exception of possibly a third-party JSON parsing library (for C++ we recommend the single-header nlohmann json). Note that for Python, NumPy is a third-party library and is not allowed. You may consult language references, but no other online or printed material. It is more important to solve fewer problems well than more problems poorly. We are looking for code quality, among other things. Before the end of your allowed time, you must email us back your code with a description of how to use it.

## Introduction

A neural network for our purposes is a composition of "building block" functions which we call _layers_. Running this function is called _inference_. In this assignment we will parse an on-disk (JSON) description of a neural network into an in-memory representation which is capable of running inference. For this problem, we will assume all inputs are two dimensional arrays (no batching, no channels).

Each layer has a name, a type, zero or more inputs, and exactly one output. Inputs and outputs are described by _tensors_: these are 2 dimensional arrays. Each layer has an on-disk JSON format from which it can be deserialized.

### ReLULayer

A `ReLULayer` is the identity function for non-negative inputs, and gives zero for negative inputs.

For example, given the input 2d tensor

    [1, -1]
    [-2, 4]

A relu layer would produce

    [1, 0]
    [0, 4]

The JSON schema of a relu layer is

    {
        "type": "ReLU",
        "name": <tring>,
        "input": <string> // the name of a layer
    }

### InputLayer

An `InputLayer` is the means of introducing data to a network. It has a fixed (2-dimensional) size and has no input. The JSON format is

    {
        "type": "Input",
        "name": some_string,
        "width": <integer>,
        "height": <integer>,
    }

### PoolingLayer

A pooling layer applies a "pooling function" to 2-dimensional sliding windows of the first two dimensions of a tensor. The pooling function for us is either "max" or "min". The windows move in _strides_ in both height and width dimensions (the first two dimensions of the tensor).

For example, given the 2d tensor:

    width direction ------------>
    [1,   2,  3,  4,  5,  6]
    [7,   8,  9, 10, 11, 12]
    [13, 14, 15, 16, 17, 18]

Max pooling with a stride of 1 and a window size of 2 gives the 2d tensor

    [ max[ 1, 2]  max[ 2, 3]        max[ 5, 6] ]
    [    [ 7, 8],    [ 8, 9], ... ,    [11,12] ]
    [                                          ] = [ 8,  9, 10, 11, 12]
    [ max[ 7, 8]  max[ 8, 9]        max[11,12] ]   [14, 15, 16, 17, 18]
    [    [13,14],    [14,15], ... ,    [17,18] ]

Min pooling with a stride of 1 and a window size of 3 gives

    [    [ 1, 2, 3]      [ 2, 3, 4]           [ 4, 5, 6] ]
    [ min[ 7, 8, 9] , min[ 8, 9,10], ... , min[10,11,12] ]  = [ 1, 2, 3, 4 ]  (width 4, height 1)
    [    [13,14,15]      [14,15,16]           [16,17,18] ]

Max pooling with a stride of 3 and a window size of 3 results in the 2d tensor

    [    [ 1, 2, 3]      [ 4, 5, 6] ]
    [ max[ 7, 8, 9] , max[10,11,12] ]  = [ 15, 18 ]  (width 2, height 1)
    [    [13,14,15]      [16,17,18] ]

The JSON schema of a `PoolingLayer` is

    {
        "type": "Pooling",
        "name": <string>,
        "stride": <integer>, // optional -- if not provided, assume 1
        "window_size": <integer>, // assume square windows
        "pooling_kind": , // one of "max" or "min"
        "input": <string>
    }

You may restrict yourself to windows which lie entirely over the input tensor. For example, if we applied max pooling to the same input from before with a window size of 3 and a stride of 4, we would only take

    [    [ 1, 2, 3] ]
    [ max[ 7, 8, 9] ]  = [ 15 ]  (width 1, height 1)
    [    [13,14,15] ]

(the next windows in the width and height dimensions reach over the edge of the input tensor)

## **PROBLEM 1**

Decide on a representation for tensors. Given an input and output tensor, write a function that computes ReLU inference and writes its result to the output parameter. What precondition on the output tensor dimensions must be met to write to it? (you may assume such preconditions in your implementation)

## **PROBLEM 2**

Decide on a representation for pooling parameters, `PoolingParameters`. Given an input tensor, an output tensor, and pooling parameters, write a function that computes Pooling inference and writes its result to the output parameter. What precondition on the output tensor dimensions must be met to write the result to it? (you may assume such preconditions in your implementation)

## **PROBLEM 3**

Using your representation of tensors and pooling parameters, design a class `Net` which can be instantiated from a JSON description of its layers. `Net` must implement 3 functions: `SetInput`, `DoInference`, and `GetOutput`. `SetInput` takes the (string) name of an `InputLayer` and a tensor (the input data). `DoInference` takes no arguments and computes the composition of all network layers. `GetOutput` takes the (string) name of a layer and gives its output. For example, given the following on-disk network:

    [
        {
            "type": "Input",
            "name": "input_name",
            "width": 4,
            "height": 2
        },
        {
            "type": "Pooling",
            "name": "pooling_name",
            "window_size": 2,
            "stride": 2,
            "pooling_kind": "max",
            "input": "input_name"
        },
        {
            "type": "ReLU",
            "name": "relu",
            "input": "pooling_name"
        }
    ]

And input tensor

    input = [-1, -2, -3, 4]
            [-5, -1,  7, 8]

if `net` represents our network,

    net.SetInput("input_name", input)
    net.DoInference()
    output = net.GetOutput("relu")

Then `output` represents the tensor

    [0, 8]    (width 2, height 1)

## **BONUS PROBLEM 4**

### ConvolutionLayer

A `ConvLayer` is similar to a `PoolingLayer`, except that instead of applying a summary pooling function on each sliding window, it computes a weighted-average of the elements (aka, a "filter"). The JSON schema of a `ConvLayer` is

    {
        "type": "Conv",
        "name": <string>,
        "stride": <integer>, // optional -- if not provided, assume 1
        "window_size": <integer>, //  assume square windows
        "weights": <list of numbers>
        "input": <string>
    }

Weights are stored in a row-major fashion, meaning

    weight[rowIdx, colIdx] = weights_list[rowIdx * window_size + colIdx]

For example, given input tensor

    input = [1, 2, 3, 4]
            [5, 6, 7, 8]

and convolving with

    {
        "type": "Conv",
        "name": "example_conv",
        "stride": 2,
        "window_size": 2,
        "weights": [x, y, z, w] // symbolic weights in this example; actual weights are numeric values
    }

gives

    [1*x + 2*y + 5*z + 6*w, 3*x + 4*y + 7*z + 8*w]

Extend your `Net` class and JSON parser to handle `ConvLayer`s. As in the `PoolingLayer`, you may restrict to sliding windows which do not extend past the edge of the input tensor.
