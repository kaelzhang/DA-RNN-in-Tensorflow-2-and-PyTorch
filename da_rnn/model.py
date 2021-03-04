import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM

from .layers import (
    EncoderInput,
    Decoder
)


class DARNN(Model):
    def __init__(self, T, m, p):
        super().__init__(name='DARNN')

        self.m = m
        self.encoder_input = EncoderInput(T=T, m=m)
        self.encoder_lstm = LSTM(m, return_sequences=True)

        self.decoder = Decoder(T=T, p=p, m=m)

    def call(self, inputs):
        """
        """

        X, dec_data = inputs
        batch_size = X.shape[0]

        h0 = tf.zeros((batch_size, self.m))
        s0 = tf.zeros((batch_size, self.m))

        x_tilde = self.encoder_input(
            X, h0, s0
        )

        # Equation 11
        encoder_h = self.encoder_lstm(x_tilde)

        y_hat_T = self.decoder(
            dec_data, encoder_h, h0, s0
        )

        return tf.squeeze(y_hat_T)
