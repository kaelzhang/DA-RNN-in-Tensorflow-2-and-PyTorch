[![](https://travis-ci.org/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch.svg?branch=master)](https://travis-ci.org/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch)
[![](https://codecov.io/gh/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch/branch/master/graph/badge.svg)](https://codecov.io/gh/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch)
[![](https://img.shields.io/pypi/v/da-rnn.svg)](https://pypi.org/project/da_rnn/)
[![](https://img.shields.io/pypi/l/da-rnn.svg)](https://github.com/kaelzhang/DA-RNN-in-Tensorflow-2-and-PyTorch)

# Tensorflow 2 DA-RNN

A Tensorflow 2 (Keras) implementation of the [Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)

Paper: [https://arxiv.org/abs/1704.02971](https://arxiv.org/abs/1704.02971)

## Install

For Tensorflow 2

```sh
pip install da-rnn[keras]
```

For PyTorch

```sh
pip install da-rnn[torch]
```

## Usage

For Tensorflow 2

```py
from da_rnn.keras import (
  DARNN
)

model = DARNN(10, 64, 64)

y_hat = model(inputs)
```

For PyTorch

```py
from da_rnn.torch import (
  DARNN
)

model = DARNN(10, 64, 64)
```

### Python Docstring Notations

In docstrings of the methods of this project, we have the following notation convention:

```
variable_{subscript}__{superscript}
```

For example:

- `y_T__i` means ![y_T__i](https://render.githubusercontent.com/render/math?math=y_T^1), the `i`-th prediction value at time `T`.
- `alpha_t__k` means ![alpha_t__k](https://render.githubusercontent.com/render/math?math=\alpha_t^k), the attention weight measuring the importance of the `k`-th input feature (driving series) at time `t`.

### DARNN(T, m, p, y_dim=1)

> The naming of the following (hyper)parameters is consistent with the paper, except `y_dim` which is not mentioned in the paper.

- **T** `int` the length (time steps) of the window
- **m** `int` the number of the encoder hidden states
- **p** `int` the number of the decoder hidden states
- **y_dim** `int=1` the prediction dimention. Defaults to `1`.

Return the DA-RNN model instance.

## Data Processing

Each feature item of the dataset should be of shape `(batch_size, T, length_of_driving_series + y_dim)`

And each label item of the dataset should be of shape `(batch_size, 1, y_dim)`

## Development

Install dependencies:

```sh
make install
```

Run notebook:

```sh
cd notebook
jupyter lab
```

## TODO
- [x] no hardcoding (`1` for now) for prediction dimentionality

## License

[MIT](LICENSE)
