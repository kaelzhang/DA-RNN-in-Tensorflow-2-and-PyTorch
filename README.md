[![](https://travis-ci.org/kaelzhang/tensorflow-2-DA-RNN.svg?branch=master)](https://travis-ci.org/kaelzhang/tensorflow-2-DA-RNN)
[![](https://codecov.io/gh/kaelzhang/tensorflow-2-DA-RNN/branch/master/graph/badge.svg)](https://codecov.io/gh/kaelzhang/tensorflow-2-DA-RNN)
[![](https://img.shields.io/pypi/v/da-rnn.svg)](https://pypi.org/project/da_rnn/)
[![](https://img.shields.io/pypi/l/da-rnn.svg)](https://github.com/kaelzhang/tensorflow-2-DA-RNN)

# Tensorflow 2 DA-RNN

A Tensorflow 2 (Keras) implementation of the [Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)

Paper: [https://arxiv.org/abs/1704.02971](https://arxiv.org/abs/1704.02971)

## Install

```sh
pip install da-rnn
```

## Usage

```py
from da_rnn import (
  DARNN
)

model = DARNN(10, 64, 64)

y_hat = model(inputs)
```

### DARNN(T, m, p, y_dim=1)

- **T** `int` the length (time steps) of the window size
- **m** `int` the number of the encoder hidden states
- **p** `int` the number of the decoder hidden states
- **y_dim** `int=1` the prediction dimention

Return the DA-RNN model instance.

## Development

Install dependencies:

```sh
make install
```

## TODO
- [x] no hardcoding (`1` for now) for prediction dimentionality

## License

[MIT](LICENSE)
