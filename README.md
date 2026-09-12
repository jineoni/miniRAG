# MiniGrad: Autodiff & Neural Network Engine

MiniGrad is a learning-oriented project to implement the core mechanics behind neural network training from scratch, using Python and NumPy. It focuses on building a small but complete reverse-mode automatic differentiation engine and the neural network components built on top of it.

## Planned Features

The following are planned and not yet implemented:

- Reverse-mode automatic differentiation
- Dynamic computation graphs
- Backpropagation
- Neural network layers and loss functions
- Optimizers
- PyTorch-based validation

## Repository Structure

```
minigrad/
├── minigrad/
│   ├── tensor.py         # Core Tensor class and autodiff graph
│   ├── ops.py            # Differentiable tensor operations
│   ├── nn/
│   │   ├── module.py      # Base Module class
│   │   ├── layers.py      # Neural network layers
│   │   ├── activations.py # Activation functions
│   │   └── losses.py      # Loss functions
│   └── optim/
│       ├── sgd.py         # SGD optimizer
│       └── adam.py        # Adam optimizer
├── examples/              # End-to-end usage examples
├── tests/                 # Unit tests
├── benchmarks/            # Performance and correctness benchmarks
└── requirements.txt
```

## Roadmap

- [ ] Tensor and computation graph
- [ ] Basic differentiable operations
- [ ] Backpropagation
- [ ] Neural network modules
- [ ] Loss functions
- [ ] Optimizers
- [ ] Training examples
- [ ] PyTorch validation and benchmarks
