from typing import Optional

import torch
from torch.nn import (
    Module,
    Linear,
    LSTM
)

from da_rnn.common import (
    check_T
)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(Module):
    n: int
    T: int
    m: int

    DEVICE = DEVICE

    def __init__(
        self,
        n: int,
        T: int,
        m: int,
        dropout
    ):
        """
        Generates the new input X_tilde for encoder

        Args:
            n (int): input size, the number of features of a single driving series
            T (int): the size (time steps) of the window
            m (int): the number of the encoder hidden states
        """

        super().__init__()

        self.n = n
        self.T = T
        self.m = m

        # Two linear layers forms a bigger linear layer
        self.WU_e = Linear(m * 2 + T, T, False)

        # Since v_e ∈ R^T, the input size is T
        self.v_e = Linear(T, 1, False)

        self.lstm = LSTM(self.n, self.m, dropout=dropout)

    def forward(self, X):
        """
        Args:
            X: the n driving (exogenous) series of shape (batch_size, T, n)

        Returns:
            The encoder hidden state of shape (T, batch_size, m)
        """

        batch_size = X.shape[0]

        hidden_state = torch.zeros(1, batch_size, self.m, device=self.DEVICE)
        cell_state = torch.zeros(1, batch_size, self.m, device=self.DEVICE)

        X_encoded = torch.zeros(self.T, batch_size, self.m, device=self.DEVICE)

        for t in range(self.T):
            # [h_t-1; s_t-1]
            hs = torch.cat((hidden_state, cell_state), 2)
            # -> (1, batch_size, m * 2)

            hs = hs.permute(1, 0, 2).repeat(1, self.n, 1)
            # -> (batch_size, n, m * 2)

            tanh = torch.tanh(
                self.WU_e(
                    torch.cat((hs, X.permute(0, 2, 1)), 2)
                    # -> (batch_size, n, m * 2 + T)
                )
            )
            # -> (batch_size, n, T)

            # Equation 8
            E = self.v_e(tanh).view(batch_size, self.n)
            # -> (batch_size, n)

            # Equation 9
            Alpha_t = torch.softmax(E, 1)
            # -> (batch_size, n)

            # Ref
            # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            # The input shape of torch LSTM should be
            # (seq_len, batch, n)
            _, (hidden_state, cell_state) = self.lstm(
                (X[:, t, :] * Alpha_t).unsqueeze(0),
                # -> (1, batch_size, n)
                (hidden_state, cell_state)
            )

            X_encoded[t] = hidden_state[0]

        return X_encoded


class Decoder(Module):
    T: int
    p: int

    DEVICE = DEVICE

    def __init__(
        self,
        T: int,
        m: int,
        p: int,
        y_dim: int,
        dropout
    ):
        """
        Calculates y_hat_T

        Args:
            T (int): the size (time steps) of the window
            m (int): the number of the encoder hidden states
            p (int): the number of the decoder hidden states
            y_dim (int): prediction dimentionality
        """

        super().__init__()

        self.T = T
        self.p = p

        self.WU_d = Linear(p * 2 + m, m, False)
        self.v_d = Linear(m, 1, False)
        self.wb_tilde = Linear(y_dim + m, 1, False)

        self.lstm = LSTM(1, p, dropout=dropout)

        self.Wb = Linear(p + m, p)
        self.vb = Linear(p, y_dim)

    def forward(self, Y, X_encoded):
        """
        Args:
            Y: prediction data of shape (batch_size, T - 1, y_dim) from time 1 to time T - 1. See Figure 1(b) in the paper
            X_encoded: encoder hidden states of shape (T, batch_size, m)

        Returns:
            y_hat_T: the prediction of shape (batch_size, y_dim)
        """

        batch_size = Y.shape[0]

        hidden_state = torch.zeros(1, batch_size, self.p, device=self.DEVICE)
        cell_state = torch.zeros(1, batch_size, self.p, device=self.DEVICE)

        for t in range(self.T - 1):
            # Equation 12
            l = self.v_d(
                torch.tanh(
                    self.WU_d(
                        torch.cat(
                            (
                                torch.cat(
                                    (hidden_state, cell_state),
                                    2
                                ).permute(1, 0, 2).repeat(1, self.T, 1),
                                # -> (batch_size, T, p * 2)

                                X_encoded.permute(1, 0, 2)
                                # -> (batch_size, T, m)
                            ),
                            2
                        )
                    )
                    # -> (batch_size, T, m * 2)
                )
                # -> (batch_size, T, m)
            ).view(batch_size, self.T)
            # -> (batch_size, T)

            # Equation 13
            Beta_t = torch.softmax(l, 1)
            # -> (batch_size, T)

            # Equation 14
            context_vector = torch.bmm(
                Beta_t.unsqueeze(1),
                # -> (batch_size, 1, T)
                X_encoded.permute(1, 0, 2)
                # -> (batch_size, T, m)
            ).squeeze(1)
            # -> (batch_size, m)

            # Equation 15
            y_tilde = self.wb_tilde(
                torch.cat((Y[:, t, :], context_vector), 1)
                # -> (batch_size, y_dim + m)
            )
            # -> (batch_size, 1)

            # Equation 16
            _, (hidden_state, cell_state) = self.lstm(
                y_tilde.unsqueeze(0),
                # -> (1, batch_size, 1)
                (hidden_state, cell_state)
            )

        # Equation 22
        y_hat_T = self.vb(
            self.Wb(
                torch.cat((hidden_state.squeeze(0), context_vector), 1)
                # -> (batch_size, p + m)
            )
            # -> (batch_size, p)
        )
        # -> (batch_size, 1)

        return y_hat_T


class DARNN(Module):
    y_dim: int

    DEVICE = DEVICE

    def __init__(
        self,
        n: int,
        T: int,
        m: int,
        p: Optional[int] = None,
        y_dim: int = 1,
        dropout=0
    ):
        """
        Args:
            n (int): input size, the number of features of a single driving series
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

        super().__init__()

        check_T(T)

        self.y_dim = y_dim

        self.encoder = Encoder(n, T, m, dropout)
        self.decoder = Decoder(T, m, p or m, y_dim, dropout)

    def forward(self, inputs):
        X, Y = torch.split(
            inputs,
            [inputs.shape[2] - self.y_dim, self.y_dim],
            dim=2
        )

        return self.decoder(Y, self.encoder(X))
