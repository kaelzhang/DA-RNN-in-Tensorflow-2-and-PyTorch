from typing import Optional

from tensorflow.keras.models import Model

from da_rnn.common import (
    check_T
)

from .layers import (
    Encoder,
    Decoder
)


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
        self.decoder = Decoder(T, m, p, y_dim=y_dim)

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
        # -> (batch_size, 1, y_dim)

        return y_hat_T

    def get_config(self):
        return {
            'T': self.T,
            'm': self.m,
            'p': self.p,
            'y_dim': self.y_dim
        }
