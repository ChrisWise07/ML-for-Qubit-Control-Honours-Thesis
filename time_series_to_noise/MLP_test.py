import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import torch

torch.manual_seed(0)


class SimpleANN(nn.Module):
    def __init__(
        self,
        num_params_to_predict: int = 9,
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

        input_size = 20

        num_nodes_in_hidden_layers = int(20 * 0.66) + 9

        self.input_to_hidden = nn.Linear(
            input_size, num_nodes_in_hidden_layers
        )
        self.hidden_to_output = nn.Linear(
            num_nodes_in_hidden_layers, num_params_to_predict
        )

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

        x1 = F.silu(self.input_to_hidden(time_series_squence))
        return self.hidden_to_output(x1)
