import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from time_series_to_noise.learnable_positional_encoder import (
    LearnablePositionalEncoding,
)

from time_series_to_noise.constants import DEVICE

torch.manual_seed(0)


class EncoderWithMLP(nn.Module):

    """
    This class implements an encoder only transformer model that is
    used for analysis of times series data. The output of the encoder is
    then fed to a neural network. The final output of
    the complex valued neural network are the noise encoding matrices.

    The encoder code borrows from a transformer tutorial
    by Ludvigssen [1]. Hyperparameter and architecture details are
    borrowed from Vaswani et al (2017) [2].

    [1] Ludvigsen, K.G.A. (2022)
    'How to make a pytorch transformer for time series forecasting'.
    Medium. Towards Data Science.
    Available at: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e (Accessed: February 10, 2023).

    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: February 10, 2023).

    """

    def __init__(
        self,
        n_heads: int,
        dropout_encoder: float,
        n_encoder_layers: int,
        d_model: int,
        dim_feedforward_encoder: int,
        seq_dim: int = 2,
        seq_len: int = 1024,
        num_params_to_predict: int = 9,
        batch_first: bool = True,
        use_activation: bool = True,
    ):
        """
        Args:

            input_size (int): number of input variables.

            batch_first (bool):
                if True, the batch dimension is the first in the input
                and output tensors

            num_noise_matrices (int):
                the number of noise matrices to predict

            noise_matrix_dim (int):
                the dimension of the noise matrices to predict where
                matrices are assumed to be square

            d_model (int):
                All sub-layers in the model produce outputs
                of dimension d_model

            n_encoder_layers (int):
                number of stacked encoder layers in the encoder

            n_heads (int): the number of attention heads

            dropout_encoder (float): the dropout rate of the encoder

            dim_feedforward_encoder (int):
                number of neurons in the hidden layer of the linear
                layer
        """

        super().__init__()

        self.num_params_to_predict = num_params_to_predict
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_activation = use_activation

        self.linear_embedding_layer = nn.Linear(
            in_features=seq_dim, out_features=d_model
        )

        self.positional_encoding_layer = LearnablePositionalEncoding(
            seq_length=seq_len, d_model=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
            activation=F.silu,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers
        )

        self.linear_layer_1 = nn.Linear(in_features=d_model, out_features=1)

        self.linear_layer_2 = nn.Linear(
            in_features=seq_len, out_features=num_params_to_predict
        )

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
                    num_noise_matrices * noise_matrix_dim**2
                ]
        """
        linear_embedded_data = self.linear_embedding_layer.forward(
            time_series_squence
        )

        positional_encoded_data = self.positional_encoding_layer.forward(
            linear_embedded_data
        )

        encoder_output = self.encoder.forward(positional_encoded_data)

        fc_head_output_1 = F.silu(
            self.linear_layer_1.forward(encoder_output).squeeze()
        )

        fc_head_output_2 = self.linear_layer_2.forward(fc_head_output_1)

        mu_output = self.apply_mu_sigmoid(fc_head_output_2)

        if self.use_activation:
            return self.custom_activation(mu_output)

        return mu_output
