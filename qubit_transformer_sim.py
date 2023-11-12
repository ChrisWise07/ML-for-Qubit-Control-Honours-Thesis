from dataclasses import dataclass
import torch

from time_series_to_noise.monte_carlo_qubit_simulation import *

from time_series_to_noise.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
)

from time_series_to_noise.utils import calculate_expectation_values

import math

from torch.utils.data import Dataset, DataLoader, random_split

from torchmetrics.functional.regression import (
    symmetric_mean_absolute_percentage_error,
)

import wandb
import datetime
import os


QUBIT_ENERGY_GAP = 12.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class N3NoiseGenerator:
    """
    Class to generate N3 noise.

    Attributes:
        number_of_noise_realisations (int):
            Number of noise realisations for a single example.
        number_of_time_steps (int):
            Number of time steps for a control pulse, i.e. the number of
            values in the control pulse sequence.
        division_factor (int):
            The division factor for the N3 noise. Controls the frequency
            of the noise.
        total_time (float):
            The total time for the noise (default 1.0).
        g (float):
            The strength of the noise (default 0.2).
    """

    number_of_noise_realisations: int = 2000
    number_of_time_steps: int = 1024
    division_factor: int = 4
    total_time: float = 1.0
    g: float = 0.2

    def __post_init__(self):
        with torch.no_grad():
            time_step = self.total_time / self.number_of_time_steps

            time_range = torch.linspace(
                time_step,
                self.total_time - time_step,
                self.number_of_time_steps,
            ).to(DEVICE)

            self.time_domain_signal = generate_time_domain_signal_for_noise(
                time_range=time_range,
                total_time=self.total_time,
            )

            self.colour_filter = generate_colour_filter_for_noise(
                number_of_time_steps=self.number_of_time_steps,
                division_factor=self.division_factor,
            )

    def generate_noise_instance(self, batch_size: int) -> torch.Tensor:
        """
        Generate N3 noise.

        Args:
            batch_size (int):
                The batch size.

        Returns:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                )
        """
        with torch.no_grad():
            return generate_non_stationary_colour_gaussian_noise_time_series(
                time_domain_signal=self.time_domain_signal,
                colour_filter=self.colour_filter,
                number_of_time_steps=self.number_of_time_steps,
                number_of_noise_realisations=self.number_of_noise_realisations,
                batch_size=batch_size,
                division_factor=self.division_factor,
                g=self.g,
            )


@dataclass
class SquaredScaledNoiseGenerator:
    """
    A class that takes a noise time series and squares and scales it.

    Attributes:
        g (float):
            The strength of the scaling.
    """

    g: float = 0.2

    def generate_noise_instance(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Squares and scales the noise.

        Args:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                )

        Returns:
            noise (torch.Tensor):
                The squared and scaled noise. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                )
        """
        with torch.no_grad():
            return self.g * noise**2


@dataclass
class CombinedNoiseGenerator:
    """
    A class the combines two noise generators, one for x and one for z
    axis.

    Attributes:
        x_noise_generator (Class):
            The noise generator for the x axis.
        z_noise_generator (Class):
            The noise generator for the z axis.
    """

    x_noise_generator: N3NoiseGenerator
    z_noise_generator: SquaredScaledNoiseGenerator

    def __post_init__(self):
        self.z_noise_generator_need_x_noise = isinstance(
            self.z_noise_generator, SquaredScaledNoiseGenerator
        )

    def generate_noise_instance(self, batch_size: int) -> torch.Tensor:
        """
        Generates the noise for the x and z axis and combines them.

        Args:
            batch_size (int):
                The batch size.

        Returns:
            noise (torch.Tensor):
                The generated noise. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                    2,
                )
        """
        x_noise = self.x_noise_generator.generate_noise_instance(
            batch_size=batch_size
        )

        z_noise = self.z_noise_generator.generate_noise_instance(noise=x_noise)

        return torch.stack((x_noise, z_noise), dim=-1)


@dataclass
class ControlPulseDistortion:
    """
    A class that distorts a control pulse.

    Attributes:
        total_time (float):
            The total time for the control pulse.
        number_of_time_steps (int):
            The number of time steps for the control pulse.
    """

    total_time: float = 1.0
    number_of_time_steps: int = 1024

    def __post_init__(self):
        self.dft_matrix_of_transfer_func = (
            create_DFT_matrix_of_LTI_transfer_func_for_signal_distortion(
                total_time=self.total_time,
                number_of_time_steps=self.number_of_time_steps,
            )
        )

    def distort_control_pulse(self, pulses: torch.Tensor) -> torch.Tensor:
        """
        Distorts the control pulse.

        Args:
            pulses (torch.Tensor):
                The control pulses. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                )

        Returns:
            distorted_pulses (torch.Tensor):
                The distorted control pulses. Expected shape:
                (
                    batch_size,
                    number_of_time_steps,
                )
        """
        batch_size = pulses.shape[0]

        return torch.clamp(
            generate_distorted_signal(
                original_signal=pulses,
                dft_matrix_of_transfer_func=self.dft_matrix_of_transfer_func.repeat(
                    batch_size, 1, 1, 1
                ),
            ),
            -1,
            1,
        )


@dataclass
class QubitSimulator:
    """
    Class to simulate the evolution of a qubit under the influence of
    a control Hamiltonian and a noise Hamiltonian.

    Attributes:
        delta_t (float):
            The time step.
        control_static_operators (torch.Tensor):
            The static operators for the control Hamiltonian. Expected
            shape:
            (
                num_control_channels,
                system_dimension,
                system_dimension,
            )
        control_dynamic_operators (torch.Tensor):
            The dynamic operators for the control Hamiltonian. Expected
            shape:
            (
                num_control_channels,
                system_dimension,
                system_dimension,
            )
        noise_dynamic_operators (torch.Tensor):
            The dynamic operators for the noise Hamiltonian. Expected
            shape:
            (
                num_noise_channels,
                system_dimension,
                system_dimension,
            )
    """

    delta_t: float
    max_amp: float
    num_qubits: int
    noise_dynamic_operators: torch.Tensor
    control_static_operators: torch.Tensor
    control_dynamic_operators: torch.Tensor
    distortion: ControlPulseDistortion

    def compute_control_unitaries_all_timesteps(
        self,
        control_pulse_time_series: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = control_pulse_time_series.shape[0]

        batched_control_operators = self.control_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        batched_static_operators = self.control_static_operators.repeat(
            batch_size, 1, 1, 1
        )

        distorted_control_pulse_time_series = (
            self.distortion.distort_control_pulse(
                pulses=control_pulse_time_series
            )
        )

        control_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=distorted_control_pulse_time_series
                * self.max_amp,
                operators_for_time_evolving_elements=batched_control_operators,
                operators_for_static_elements=batched_static_operators,
            )
        )

        exponentiated_scaled_hamiltonians_ctrl = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=control_hamiltonian,
                delta_T=self.delta_t,
            )
        )

        return compute_unitaries_for_all_time_steps(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_ctrl,
        )

    def compute_vo_operators_with_learnt_noise(
        self,
        control_unitaries_all_time_steps: torch.Tensor,
        learnt_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the evolution of a qubit under the influence of a
        control as described by the control pulse time series and a
        noise Hamiltonian. Returns the VO operators and the final
        control unitaries.

        Args:
            control_pulse_time_series (torch.Tensor):
                The time series representing the pulses suggested by the
                model. Expected shape:
                (
                    batch_size,
                    num_time_steps,
                    num_control_channels
                )
        Returns:
            vo_operators (torch.Tensor):
                The VO operators. Expected shape:
                (
                    3,
                    batch_size,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
            final_control_unitaries (torch.Tensor):
                The final time step control unitaries. Expected shape:
                (
                    batch_size,
                    system_dimension,
                    system_dimension,
                )
        """
        batch_size = learnt_noise.shape[0]

        batched_noise_operators = self.noise_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        noise_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=learnt_noise,
                operators_for_time_evolving_elements=batched_noise_operators,
            )
        )

        interaction_hamiltonian = create_interaction_hamiltonian_for_each_timestep_noise_relisation_batchwise(
            control_unitaries=control_unitaries_all_time_steps,
            noise_hamiltonians=noise_hamiltonian,
        )

        exponentiated_scaled_hamiltonians_interaction = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=interaction_hamiltonian,
                delta_T=self.delta_t,
            )
        )

        final_timestep_interaction_unitaries = compute_unitary_at_timestep(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_interaction,
        )

        return construct_vo_operator_for_batch(
            final_step_control_unitaries=control_unitaries_all_time_steps[
                :, -1
            ],
            final_step_interaction_unitaries=final_timestep_interaction_unitaries,
        )

    def compute_expectations(
        self,
        Vo_operators: torch.Tensor,
        final_control_unitaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the expectation values given the VO operators and the
        final control unitaries. If the final control unitaries are not
        provided, the lastest simulation final control unitaries are
        used.

        Args:
            Vo_operators (torch.Tensor):
                The VO operators. Expected shape:
                (
                    3,
                    batch_size,
                    num_time_steps,
                    system_dimension,
                    system_dimension,
                )
            final_control_unitaries (torch.Tensor):
                The final control unitaries. Expected shape:
                (
                    batch_size,
                    system_dimension,
                    system_dimension,
                )

        Returns:
            expectations (torch.Tensor):
                The expectations. Expected shape:
                (
                    batch_size,
                    num_vo_operators * 6 ^ num_qubits
                )
        """
        return calculate_expectation_values(
            Vo_operators=Vo_operators,
            control_unitaries=final_control_unitaries,
        )


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape  [
                batch_size,
                seq_len,
                d_model
            ]
        """
        x = x + self.pe.squeeze(1)[None, :, :]
        return self.dropout(x)


class EncoderOnlyTransformerForNoise(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        dropout_encoder: float,
        n_encoder_layers: int,
        d_model: int,
        dim_feedforward_encoder: int,
        seq_len: int = 1024,
        channels_in: int = 2,
        channels_out: int = 2,
        batch_first: bool = True,
        positional_dropout: float = 0.0,
        device: torch.device = DEVICE,
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

        self.d_model = d_model
        self.seq_len = seq_len
        self.device = device
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.embedding_layer = torch.nn.Linear(
            in_features=channels_in, out_features=d_model
        )

        self.positional_encoding_layer = PositionalEncoding(
            d_model=d_model, max_len=seq_len, dropout=positional_dropout
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first,
            activation=torch.nn.SiLU(),
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers
        )

        self.encoder_output_norm = torch.nn.BatchNorm1d(num_features=d_model)

        self.linear_layer = torch.nn.Linear(
            in_features=d_model, out_features=channels_out
        )

    def forward(
        self,
        size_of_batch: int,
        num_noise_realisations: int,
    ) -> torch.Tensor:
        """
        Returns a tensor of shape:


        Args:
            size_of_batch (int): the batch size
            num_noise_realisations (int):
                the number of noise realisations

        Returns:
            Tensor:
                a tensor of shape: [batch_size, seq_len, channels_out]
        """
        white_noise = torch.randn(
            size_of_batch * num_noise_realisations,
            self.seq_len,
            self.channels_in,
        ).to(self.device)

        embedded_data = self.embedding_layer.forward(white_noise)

        positional_encoded_data = self.positional_encoding_layer.forward(
            embedded_data
        )

        encoder_output = self.encoder.forward(positional_encoded_data)

        # pytorch expects (batch, features, seq_len) for batchnorm1d
        encoder_output_transposed = encoder_output.transpose(-1, -2)

        norm_encoder_output_transposed = self.encoder_output_norm.forward(
            encoder_output_transposed
        )

        norm_encoder_output = norm_encoder_output_transposed.transpose(-1, -2)

        model_output = self.linear_layer.forward(norm_encoder_output)

        model_output_reshaped = model_output.reshape(
            size_of_batch,
            num_noise_realisations,
            self.seq_len,
            self.channels_out,
        ).transpose(1, 2)

        return model_output_reshaped


class ControlPulseDataset(Dataset):
    def __init__(self, data, targets):
        """
        Args:
            data (Tensor): A tensor containing the control pulses data.
            targets (Tensor): A tensor containing the distorted control pulses data.
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): The index of the item.

        Returns:
            A tuple with the control pulse and its corresponding distorted control pulse.
        """
        return self.data[idx], self.targets[idx]


class QubitSimWrapper:
    def __init__(self, qubit_sim):
        self.qubit_sim = qubit_sim

    def simulate_noise_with_control_pulses(
        self,
        noise: torch.Tensor,
        control_pulse_time_series: torch.Tensor,
    ) -> torch.Tensor:
        control_unitaries_all_timesteps = (
            self.qubit_sim.compute_control_unitaries_all_timesteps(
                control_pulse_time_series
            )
        )

        vo_operators = self.qubit_sim.compute_vo_operators_with_learnt_noise(
            control_unitaries_all_time_steps=control_unitaries_all_timesteps,
            learnt_noise=noise,
        )

        return self.qubit_sim.compute_expectations(
            Vo_operators=vo_operators,
            final_control_unitaries=control_unitaries_all_timesteps[:, -1],
        )


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: torch.nn.MSELoss,
    device: torch.device,
    qubit_sim_wrapper: QubitSimWrapper,
    num_noise_realisations: int,
    lrs: torch.optim.lr_scheduler.CosineAnnealingLR,
):
    model.train()
    total_loss = 0
    for _, args in enumerate(train_loader):
        optimiser.zero_grad()

        batch_control_pulses, batch_ground_truth_expectations = [
            arg.to(device) for arg in args
        ]

        model_noise = model(
            size_of_batch=batch_control_pulses.shape[0],
            num_noise_realisations=num_noise_realisations,
        )

        expectations = qubit_sim_wrapper.simulate_noise_with_control_pulses(
            noise=model_noise,
            control_pulse_time_series=batch_control_pulses,
        )

        loss = criterion.forward(expectations, batch_ground_truth_expectations)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    lrs.step()
    wandb.log({"avg_train_loss": total_loss / len(train_loader)})


def validate_model_and_save_best(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    best_val_loss: float,
    qubit_sim_wrapper: QubitSimWrapper,
    num_noise_realisations: int,
    model_path: str,
) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for args in val_loader:
            batch_control_pulses, batch_ground_truth_expectations = [
                arg.to(device) for arg in args
            ]

            model_noise = model(
                size_of_batch=batch_control_pulses.shape[0],
                num_noise_realisations=num_noise_realisations,
            )

            expectations = (
                qubit_sim_wrapper.simulate_noise_with_control_pulses(
                    noise=model_noise,
                    control_pulse_time_series=batch_control_pulses,
                )
            )

            loss = criterion(expectations, batch_ground_truth_expectations)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    wandb.log({"avg_val_loss": avg_loss})

    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model with MSE: {best_val_loss}")

    return best_val_loss


def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    qubit_sim_wrapper: QubitSimWrapper,
    num_noise_realisations: int,
):
    model.eval()
    smape_total = 0
    with torch.no_grad():
        for args in test_loader:
            batch_control_pulses, batch_ground_truth_expectations = [
                arg.to(device) for arg in args
            ]

            model_noise = model(
                size_of_batch=batch_control_pulses.shape[0],
                num_noise_realisations=num_noise_realisations,
            )

            expectations = (
                qubit_sim_wrapper.simulate_noise_with_control_pulses(
                    noise=model_noise,
                    control_pulse_time_series=batch_control_pulses,
                )
            )

            smape = symmetric_mean_absolute_percentage_error(
                expectations, batch_ground_truth_expectations
            )
            smape_total += smape

    smape_score = 1 - 0.5 * smape_total / len(test_loader)
    print(f"Test SMAPE: {smape_score}")
    wandb.log({"test_smape": smape_score})
    return smape_score


def main():
    max_amp = 100
    num_qubits = 1
    number_of_noise_realisations = 1000
    sequence_length = 64
    total_time = 1.0
    num_epochs = 200
    train_batch_size = 64
    learning_rate = 0.01

    control_static_operators = 0.5 * QUBIT_ENERGY_GAP * SIGMA_Z

    control_dynamic_operators = torch.stack(
        (0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0
    )

    noise_dynamic_operators = torch.stack((SIGMA_X, SIGMA_Y, SIGMA_Z), dim=0)

    control_pulse_distortion = ControlPulseDistortion(
        total_time=total_time,
        number_of_time_steps=sequence_length,
    )

    sim = QubitSimulator(
        delta_t=total_time / sequence_length,
        max_amp=max_amp,
        num_qubits=num_qubits,
        noise_dynamic_operators=noise_dynamic_operators,
        control_static_operators=control_static_operators,
        control_dynamic_operators=control_dynamic_operators,
        distortion=control_pulse_distortion,
    )

    qubit_sim_wrapper = QubitSimWrapper(sim)

    transformer_for_noise = EncoderOnlyTransformerForNoise(
        n_heads=2,
        dropout_encoder=0.05,
        positional_dropout=0.05,
        n_encoder_layers=3,
        d_model=16,
        dim_feedforward_encoder=16,
        seq_len=sequence_length,
        channels_in=3,
        channels_out=3,
    ).to(DEVICE)

    random_control_pulses = torch.load(
        f"random_control_pulses_{sequence_length}.pt"
    )

    ground_turth_expectations = torch.load(
        f"expectations_{sequence_length}.pt"
    )

    control_pulse_dataset = ControlPulseDataset(
        data=random_control_pulses, targets=ground_turth_expectations
    )

    num_examples = random_control_pulses.shape[0]

    train_dataset, val_dataset = random_split(
        control_pulse_dataset,
        [int(0.9 * num_examples), int(0.1 * num_examples)],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size // 2,
        shuffle=False,
        num_workers=0,
    )

    criterion = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(
        transformer_for_noise.parameters(), lr=learning_rate
    )

    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs
    )

    best_val_loss = float("inf")
    heartbeat = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_path = f"{heartbeat}_best_transformer_for_noise_{sequence_length}.pt"
    os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        name=f"transformer_for_noise_{sequence_length}_{heartbeat}",
        project="honours-research",
        config={
            "sequence_length": sequence_length,
            "max_amp": max_amp,
            "num_qubits": num_qubits,
            "number_of_noise_realisations": number_of_noise_realisations,
            "total_time": total_time,
            "num_epochs": num_epochs,
            "train_batch_size": train_batch_size,
            "n_heads": 2,
            "dropout_encoder": 0.05,
            "positional_dropout": 0.05,
            "n_encoder_layers": 3,
            "d_model": 16,
            "dim_feedforward_encoder": 16,
            "channels_in": 3,
            "channels_out": 3,
            "optimiser": "Adam",
            "lr": learning_rate,
        },
    )

    for _ in range(num_epochs):
        train_epoch(
            model=transformer_for_noise,
            train_loader=train_loader,
            optimiser=optimiser,
            criterion=criterion,
            device=DEVICE,
            qubit_sim_wrapper=qubit_sim_wrapper,
            num_noise_realisations=number_of_noise_realisations,
            lrs=lrs,
        )

        best_val_loss = validate_model_and_save_best(
            model=transformer_for_noise,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            best_val_loss=best_val_loss,
            qubit_sim_wrapper=qubit_sim_wrapper,
            num_noise_realisations=number_of_noise_realisations,
            model_path=model_path,
        )

    transformer_for_noise.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    print("Loaded best transformer_for_noise for testing.")

    smape_score = test_model(
        model=transformer_for_noise,
        test_loader=val_loader,
        device=DEVICE,
        qubit_sim_wrapper=qubit_sim_wrapper,
        num_noise_realisations=2 * number_of_noise_realisations,
    )

    print(f"Test SMAPE: {smape_score}")


if __name__ == "__main__":
    main()
