import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from time_series_to_noise.constants import DEVICE

torch.manual_seed(0)


class SimpleANN(nn.Module):
    def __init__(
        self,
        seq_len: int = 1024,
        seq_dim: int = 2,
        num_params_to_predict: int = 9,
        use_activation: bool = True,
    ):
        """
        Args:

            input_size (int): the length of the sequence.

            num_noise_matrices (int):
                the number of noise matrices to predict

            noise_matrix_dim (int):
                the dimension of the noise matrices to predict where
                matrices are assumed to be square

        """

        super().__init__()

        self.num_params_to_predict = num_params_to_predict
        self.use_activation = use_activation
        input_size = seq_len * seq_dim

        if seq_dim > 2:
            self.params_to_time_series = nn.Linear(input_size, 512)
            input_size = 512

        self.input_to_h1 = nn.Linear(input_size, 10)
        self.h1_to_h2 = nn.Linear(10, 180)
        self.h2_to_output = nn.Linear(180, num_params_to_predict)

        if num_params_to_predict == 9:
            self.psi_indices = torch.tensor([0, 3, 6])
            self.theta_indices = torch.tensor([1, 4, 7])
            self.mu_indices = torch.tensor([2, 5, 8])
        elif num_params_to_predict == 12:
            self.psi_indices = torch.tensor([0, 4, 8])
            self.theta_indices = torch.tensor([1, 5, 9])
            self.delta_indices = torch.tensor([2, 6, 10])
            self.mu_indices = torch.tensor([3, 7, 11])

    def custom_activation(self, x: Tensor) -> Tensor:
        x[:, self.psi_indices] = (
            torch.tanh(x[:, self.psi_indices]) * torch.pi / 2
        )
        x[:, self.theta_indices] = (
            torch.sigmoid(x[:, self.theta_indices]) * torch.pi / 2
        )

        if self.num_params_to_predict == 12:
            x[:, self.delta_indices] = (
                torch.tanh(x[:, self.delta_indices]) * torch.pi / 2
            )
            return x

        return x

    def apply_mu_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.mu_indices] = torch.sigmoid(x[:, self.mu_indices])
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
                    num_noise_matrices * (noise_matrix_dim**2)
                ]
        """
        batch_size, seq_len, feature_dim = time_series_squence.shape

        time_series_squence = time_series_squence.reshape(
            batch_size, seq_len * feature_dim
        )

        if feature_dim > 2:
            time_series_squence = F.silu(
                self.params_to_time_series.forward(time_series_squence)
            )

        h1 = F.silu(self.input_to_h1.forward(time_series_squence))
        h2 = F.silu(self.h1_to_h2.forward(h1))
        output = self.h2_to_output.forward(h2)

        mu_activation = self.apply_mu_sigmoid(output)

        if self.use_activation:
            return self.custom_activation(mu_activation)

        return mu_activation
