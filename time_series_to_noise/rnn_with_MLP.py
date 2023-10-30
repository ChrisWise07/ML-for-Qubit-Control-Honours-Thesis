import torch.nn as nn
from torch import nn, Tensor
import torch

torch.manual_seed(0)


class RNNWithMLP(nn.Module):
    def __init__(
        self,
        num_params_to_predict: int = 12,
        batch_first: bool = True,
        gru1_hidden_size: int = 10,
        gru2_hidden_size: int = 60,
        seq_dim: int = 2,
        use_activation: bool = True,
    ):
        """
        Args:

            input_size (int): number of input variables.

            num_params_to_predict (int):
                the number of parameters to predict

            batch_first (bool):
                if True, the batch dimension is the first in the input
                and output tensors

            gru1_hidden_size (int):
                the number of hidden units in the first GRU layer

            gru2_hidden_size (int):
                the number of hidden units in the second GRU layer
        """

        super().__init__()

        self.num_params_to_predict = num_params_to_predict
        self.use_activation = use_activation

        self.gru1 = nn.GRU(
            input_size=seq_dim,
            hidden_size=gru1_hidden_size,
            batch_first=batch_first,
        )

        self.gru2x = nn.GRU(
            input_size=gru1_hidden_size,
            hidden_size=gru2_hidden_size,
            batch_first=batch_first,
        )

        self.gru2y = nn.GRU(
            input_size=gru1_hidden_size,
            hidden_size=gru2_hidden_size,
            batch_first=batch_first,
        )

        self.gru2z = nn.GRU(
            input_size=gru1_hidden_size,
            hidden_size=gru2_hidden_size,
            batch_first=batch_first,
        )

        linear_output_size = int(num_params_to_predict / 3)
        self.fcx = nn.Linear(gru2_hidden_size, linear_output_size)
        self.fcy = nn.Linear(gru2_hidden_size, linear_output_size)
        self.fcz = nn.Linear(gru2_hidden_size, linear_output_size)

    def custom_activation(self, x: Tensor) -> Tensor:
        x[:, 0] = torch.tanh(x[:, 0]) * torch.pi / 2
        x[:, 1] = torch.sigmoid(x[:, 1]) * torch.pi / 2

        if self.num_params_to_predict == 12:
            x[:, 2] = torch.tanh(x[:, 2]) * torch.pi / 2
            return x

        return x

    def apply_mu_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_params_to_predict == 12:
            x[:, 3] = torch.sigmoid(x[:, 3])
            return x

        x[:, 2] = torch.sigmoid(x[:, 2])
        return x

    def forward(
        self,
        time_series_squence: Tensor,
    ) -> Tensor:
        """
        Returns a tensor of shape:


        Args:
            time_series_squence (Tensor):
                a tensor of shape [batch_size, seq_len]

        Returns:
            Tensor:
                a tensor of shape: [
                    batch_size,
                    num_params_to_predict
                ]
        """
        out, _ = self.gru1.forward(time_series_squence)

        gru_outx, _ = self.gru2x.forward(out)
        gru_outy, _ = self.gru2y.forward(out)
        gru_outz, _ = self.gru2z.forward(out)

        x = self.fcx(gru_outx[:, -1, :])
        y = self.fcy(gru_outy[:, -1, :])
        z = self.fcz(gru_outz[:, -1, :])

        x_mu_activation = self.apply_mu_sigmoid(x)
        y_mu_activation = self.apply_mu_sigmoid(y)
        z_mu_activation = self.apply_mu_sigmoid(z)

        if self.use_activation:
            return torch.cat(
                [
                    self.custom_activation(x_mu_activation),
                    self.custom_activation(y_mu_activation),
                    self.custom_activation(z_mu_activation),
                ],
                dim=1,
            )

        return torch.cat(
            [x_mu_activation, y_mu_activation, z_mu_activation], dim=1
        )
