import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Layer,
    LSTM,
    Dense,
    RepeatVector,
    Permute
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

        # Equation 8:
        e = self.v_e(
            tf.math.tanh(
                self.W_e(
                    # [h_t-1; s_t-1]
                    RepeatVector(n)(
                        tf.concat([hidden_state, cell_state], axis=-1)
                        # -> (batch_size, m * 2)
                    )
                    # -> (batch_size, n, m * 2)
                )
                # -> (batch_size, n, T)

                + self.U_e(
                    Permute((2, 1))(X)
                    # -> (batch_size, n, T)
                )
                # -> (batch_size, n, T)
            )
            # -> (batch_size, n, T)
        )
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


class EncoderInput(Layer):
    T: int

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

        self.initial_state = None

    def call(
        self,
        X,
        h0,
        s0
    ):
        """
        Args:
            X: the n driving (exogenous) series of shape (batch_size, T, n)
            h0: the initial encoder hidden state
            s0: the initial encoder cell state

        Returns:
            The new input (x_tilde_1, ..., x_tilde_t, ..., x_tilde_T)
        """

        alpha_weights = []

        for t in range(self.T):
            x = X[:, None, t, :]
            # -> (batch_size, n) -> (batch_size, 1, n)

            hidden_state, _, cell_state = self.input_lstm(
                x,
                initial_state=[h0, s0]
            )

            Alpha_t = self.input_attention(hidden_state, cell_state, X)
            # -> (batch_size, 1, n)

            alpha_weights.append(Alpha_t)

        # Equation 10
        return tf.multiply(X, tf.concat(alpha_weights, axis=1))
        # -> (batch_size, T, n)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'm': self.m
        })
        return config


class TemporalAttention(Layer):
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
        encoder_h
    ):
        """
        Args:
            hidden_state: hidden state `d` of shape (batch_size, p)
            cell_state: cell state `s` of shape (batch_size, p)
            encoder_h: the encoder hidden states (batch_size, T, m)

        Returns:
            The attention weights for encoder hidden states (beta_t)
        """

        # Equation 12
        l = self.v_d(
            tf.math.tanh(
                self.W_d(
                    RepeatVector(encoder_h.shape[1])(
                        tf.concat([hidden_state, cell_state], axis=-1)
                        # -> (batch_size, p * 2)
                    )
                    # -> (batch_size, T, p * 2)
                )
                # -> (batch_size, T, m)
                + self.U_d(encoder_h)
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
        self.encoder_lstm_units = m

        self.dense_Wb = Dense(p)
        self.dense_vb = Dense(y_dim)

    def call(
        self,
        Y,
        encoder_h,
        h0,
        s0
    ):
        """
        Args:
            Y: prediction data of shape (batch_size, T - 1, y_dim) from time 1 to time T - 1. See Figure 1(b) in the paper
            encoder_h: encoder hidden states of shape (batch_size, T, m)
            h0: initial decoder hidden state
            s0: initial decoder cell state
        """

        hidden_state = None
        batch_size = K.shape(encoder_h)[0]

        # c in the paper
        context_vector = tf.zeros((batch_size, 1, self.encoder_lstm_units))
        # -> (batch_size, 1, m)

        for t in range(self.T - 1):
            y = Y[:, None, t, :]
            # -> (batch_size, 1, y_dim)

            # Equation 15
            y_tilde = self.dense(
                tf.concat([y, context_vector], axis=-1)
                # -> (batch_size, 1, y_dim + m)
            )
            # -> (batch_size, 1, 1)

            # Equation 16
            hidden_state, _, cell_state = self.decoder_lstm(
                y_tilde,
                initial_state=[h0, s0]
            )
            # -> (batch_size, p)

            Beta_t = self.temp_attention(
                hidden_state,
                cell_state,
                encoder_h
            )
            # -> (batch_size, T, 1)

            # Equation 14
            context_vector = tf.matmul(
                Beta_t, encoder_h, transpose_a=True
            )
            # -> (batch_size, 1, m)

        concatenated = tf.concat(
            [hidden_state[:, None, :], context_vector], axis=-1
        )
        # -> (batch_size, 1, m + p)

        # Equation 22
        return self.dense_vb(
            self.dense_Wb(concatenated)
            # -> (batch_size, 1, p)
        )
        # -> (batch_size, 1, y_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'T': self.T,
            'm': self.m,
            'p': self.p,
            'y_dim': self.y_dim
        })
        return config
