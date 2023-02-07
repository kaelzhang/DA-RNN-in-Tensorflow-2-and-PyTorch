from typing import Optional

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Layer,
    LSTM,
    Dense,
    Permute
)

from tensorflow.keras.models import Model

from da_rnn.common import (
    check_T
)


"""
Notation (according to the paper)

Naming Convention::

    Variable_{time_step}__{sequence_number_of_driving_series}

Variables / HyperParameters:
    T (int): the size (time steps) of the window
    m (int): the number of the encoder hidden states
    p (int): the number of the decoder hidden states
    n (int): the number of features of a single driving series
    X: the n driving (exogenous) series of shape (batch_size, T, n)
    X_tilde: the new input for the encoder, i.e. X̃ = (x̃_1, ..., x̃_t, x̃_T)
    Y: the historical/previous T - 1 predictions, (y_1, y_2, ..., y_Tminus1)

    hidden_state / h: hidden state
    cell_state / s: cell state
    Alpha_t: attention weights of the input attention layer at time t
    Beta_t: attention weights of the temporal attention layer at time t
"""


class InputAttention(Layer):
    T: int

    def __init__(self, T, **kwargs):
        """
        Calculates the encoder attention weight Alpha_t at time t

        Args:
            T (int): the size (time steps) of the window
        """

        super().__init__(name='input_attention', **kwargs)

        self.T = T

        self.W_e = Dense(T)
        self.U_e = Dense(T)
        self.v_e = Dense(1)

    def call(
        self,
        hidden_state,
        cell_state,
        X
    ):
        """
        Args:
            hidden_state: hidden state of shape (batch_size, m) at time t - 1
            cell_state: cell state of shape (batch_size, m) at time t - 1
            X: the n driving (exogenous) series of shape (batch_size, T, n)

        Returns:
            The attention weights (Alpha_t) at time t, i.e.
            (a_t__1, a_t__2, ..., a_t__n)
        """

        n = X.shape[2]

        # [h_t-1; s_t-1]
        hs = K.repeat(
            tf.concat([hidden_state, cell_state], axis=-1),
            # -> (batch_size, m * 2)
            n
        )
        # -> (batch_size, n, m * 2)

        tanh = tf.math.tanh(
            tf.concat([
                self.W_e(hs),
                # -> (batch_size, n, T)

                self.U_e(
                    Permute((2, 1))(X)
                    # -> (batch_size, n, T)
                ),
                # -> (batch_size, n, T)
            ], axis=-1)
            # -> (batch_size, n, T * 2)
        )
        # -> (batch_size, n, T * 2)

        # Equation 8:
        e = self.v_e(tanh)
        # -> (batch_size, n, 1)

        # Equation: 9
        return tf.nn.softmax(
            Permute((2, 1))(e)
            # -> (batch_size, 1, n)
        )
        # -> (batch_size, 1, n)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T
        })
        return config


class Encoder(Layer):
    T: int
    m: int

    def __init__(
        self,
        T: int,
        m: int,
        **kwargs
    ):
        """
        Generates the new input X_tilde for encoder

        Args:
            T (int): the size (time steps) of the window
            m (int): the number of the encoder hidden states
        """

        super().__init__(name='encoder_input', **kwargs)

        self.T = T
        self.m = m

        self.input_lstm = LSTM(m, return_state=True)
        self.input_attention = InputAttention(T)

    def call(self, X) -> tf.Tensor:
        """
        Args:
            X: the n driving (exogenous) series of shape (batch_size, T, n)

        Returns:
            The encoder hidden state of shape (batch_size, T, m)
        """

        batch_size = K.shape(X)[0]

        hidden_state = tf.zeros((batch_size, self.m))
        cell_state = tf.zeros((batch_size, self.m))

        X_encoded = []

        for t in range(self.T):
            Alpha_t = self.input_attention(hidden_state, cell_state, X)

            # Equation 10
            X_tilde_t = tf.multiply(
                Alpha_t,
                # TODO:
                # make sure it can share the underlying data
                X[:, None, t, :]
            )
            # -> (batch_size, 1, n)

            # Equation 11
            hidden_state, _, cell_state = self.input_lstm(
                X_tilde_t,
                initial_state=[hidden_state, cell_state]
            )

            X_encoded.append(
                hidden_state[:, None, :]
                # -> (batch_size, 1, m)
            )

        return tf.concat(X_encoded, axis=1)
        # -> (batch_size, T, m)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'm': self.m
        })
        return config


class TemporalAttention(Layer):
    m: int

    def __init__(self, m: int, **kwargs):
        """
        Calculates the attention weights::

            Beta_t = (beta_t__1, ..., beta_t__i, ..., beta_t__T) (1 <= i <= T)

        for each encoder hidden state h_t at the time step t

        Args:
            m (int): the number of the encoder hidden states
        """

        super().__init__(name='temporal_attention', **kwargs)

        self.m = m

        self.W_d = Dense(m)
        self.U_d = Dense(m)
        self.v_d = Dense(1)

    def call(
        self,
        hidden_state,
        cell_state,
        X_encoded
    ):
        """
        Args:
            hidden_state: hidden state `d` of shape (batch_size, p)
            cell_state: cell state `s` of shape (batch_size, p)
            X_encoded: the encoder hidden states (batch_size, T, m)

        Returns:
            The attention weights for encoder hidden states (beta_t)
        """

        # Equation 12
        l = self.v_d(
            tf.math.tanh(
                tf.concat([
                    self.W_d(
                        K.repeat(
                            tf.concat([hidden_state, cell_state], axis=-1),
                            # -> (batch_size, p * 2)
                            X_encoded.shape[1]
                        )
                        # -> (batch_size, T, p * 2)
                    ),
                    # -> (batch_size, T, m)
                    self.U_d(X_encoded)
                ], axis=-1)
                # -> (batch_size, T, m * 2)
            )
            # -> (batch_size, T, m)
        )
        # -> (batch_size, T, 1)

        # Equation 13
        return tf.nn.softmax(l, axis=1)
        # -> (batch_size, T, 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'm': self.m
        })
        return config


class Decoder(Layer):
    T: int
    m: int
    p: int
    y_dim: int

    def __init__(
        self,
        T: int,
        m: int,
        p: int,
        y_dim: int,
        **kwargs
    ):
        """
        Calculates y_hat_T

        Args:
            T (int): the size (time steps) of the window
            m (int): the number of the encoder hidden states
            p (int): the number of the decoder hidden states
            y_dim (int): prediction dimentionality
        """

        super().__init__(name='decoder', **kwargs)

        self.T = T
        self.m = m
        self.p = p
        self.y_dim = y_dim

        self.temp_attention = TemporalAttention(m)
        self.dense = Dense(1)
        self.decoder_lstm = LSTM(p, return_state=True)

        self.Wb = Dense(p)
        self.vb = Dense(y_dim)

    def call(self, Y, X_encoded) -> tf.Tensor:
        """
        Args:
            Y: prediction data of shape (batch_size, T - 1, y_dim) from time 1 to time T - 1. See Figure 1(b) in the paper
            X_encoded: encoder hidden states of shape (batch_size, T, m)

        Returns:
            y_hat_T: the prediction of shape (batch_size, y_dim)
        """

        batch_size = K.shape(X_encoded)[0]
        hidden_state = tf.zeros((batch_size, self.p))
        cell_state = tf.zeros((batch_size, self.p))

        # c in the paper
        context_vector = tf.zeros((batch_size, 1, self.m))
        # -> (batch_size, 1, m)

        for t in range(self.T - 1):
            Beta_t = self.temp_attention(
                hidden_state,
                cell_state,
                X_encoded
            )
            # -> (batch_size, T, 1)

            # Equation 14
            context_vector = tf.matmul(
                Beta_t, X_encoded, transpose_a=True
            )
            # -> (batch_size, 1, m)

            # Equation 15
            y_tilde = self.dense(
                tf.concat([Y[:, None, t, :], context_vector], axis=-1)
                # -> (batch_size, 1, y_dim + m)
            )
            # -> (batch_size, 1, 1)

            # Equation 16
            hidden_state, _, cell_state = self.decoder_lstm(
                y_tilde,
                initial_state=[hidden_state, cell_state]
            )
            # -> (batch_size, p)

        concatenated = tf.concat(
            [hidden_state[:, None, :], context_vector], axis=-1
        )
        # -> (batch_size, 1, m + p)

        # Equation 22
        y_hat_T = self.vb(
            self.Wb(concatenated)
            # -> (batch_size, 1, p)
        )
        # -> (batch_size, 1, y_dim)

        return tf.squeeze(y_hat_T, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'm': self.m,
            'p': self.p,
            'y_dim': self.y_dim
        })
        return config


class DARNN(Model):
    def __init__(
        self,
        T: int,
        m: int,
        p: Optional[int] = None,
        y_dim: int = 1
    ):
        """
        Args:
            T (int): the size (time steps) of the window
            m (int): the number of the encoder hidden states
            p (:obj:`int`, optional): the number of the decoder hidden states. Defaults to `m`
            y_dim (:obj:`int`, optional): prediction dimentionality. Defaults to `1`

        Model Args:
            inputs: the concatenation of
            - n driving series (x_1, x_2, ..., x_T) and
            - the previous (historical) T - 1 predictions (y_1, y_2, ..., y_Tminus1, zero)

        `inputs` Explanation::

            inputs_t = (x_t__1, x_t__2, ..., x_t__n, y_t__1, y_t__2, ..., y_t__d)

            where
            - d is the prediction dimention
            - y_T__i = 0, 1 <= i <= d.

            Actually, the model will not use the value of y_T

        Usage::

            model = DARNN(10, 64, 64)
            y_hat = model(inputs)
        """

        super().__init__(name='DARNN')

        check_T(T)

        self.T = T
        self.m = m
        self.p = p or m
        self.y_dim = y_dim

        self.encoder = Encoder(T, m)
        self.decoder = Decoder(T, m, self.p, y_dim=y_dim)

    # Equation 1
    def call(self, inputs):
        X = inputs[:, :, :-self.y_dim]
        # -> (batch_size, T, n)

        # Y's window size is one less than X's
        # so, abandon `y_T`

        # By doing this, there are some benefits which makes it pretty easy to
        # process datasets
        Y = inputs[:, :, -self.y_dim:]
        # -> (batch_size, T - 1, y_dim)

        X_encoded = self.encoder(X)

        y_hat_T = self.decoder(Y, X_encoded)
        # -> (batch_size, y_dim)

        return y_hat_T

    def get_config(self):
        return {
            'T': self.T,
            'm': self.m,
            'p': self.p,
            'y_dim': self.y_dim
        }
