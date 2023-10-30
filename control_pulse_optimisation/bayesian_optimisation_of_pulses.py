from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import numpy as np
from typing import Tuple, Dict

import torch
from typing import List

from time_series_to_noise.monte_carlo_qubit_simulation import *

from time_series_to_noise.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    SIGMA_I,
    UNIVERSAL_GATE_SET_SINGLE_QUBIT,
)

from time_series_to_noise.utils import (
    calculate_expectation_values,
    fidelity_batch_of_matrices,
    compute_nuc_norm_of_diff_between_batch_of_matrices,
)

from bayes_opt_utils import (
    expected_improvement_for_minimising_obj_func,
)

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

torch.manual_seed(0)
np.random.seed(0)

QUBIT_ENERGY_GAP = 12.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TIME = 1.0
MAX_AMP = 100
NUM_QUBITS = 1
NUMBER_OF_NOISE_RELISATIONS = 2000 
NUMBER_OF_CONTROL_CHANELS = 2 if NUM_QUBITS == 1 else 5
MAX_INITIAL_PULSES = 50
MAX_SEQUENCE_LENGTH = 1024
MAX_ITERATIONS = 100


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
            time_step = 0.5 * self.total_time / self.number_of_time_steps

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
    MAX_AMP: float
    num_qubits: int
    noise_dynamic_operators: torch.Tensor
    control_static_operators: torch.Tensor
    control_dynamic_operators: torch.Tensor
    noise_generator: CombinedNoiseGenerator
    ideal_matrices: Optional[torch.Tensor] = None

    def change_ideal_matrices(self, new_ideal_matrices: torch.Tensor):
        self.ideal_matrices = new_ideal_matrices

    def compute_vo_operators_and_final_control_unitaries(
        self,
        control_pulse_time_series: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        batch_size = control_pulse_time_series.shape[0]

        batched_control_operators = self.control_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        batched_noise_operators = self.noise_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        batched_static_operators = self.control_static_operators.repeat(
            batch_size, 1, 1, 1
        )

        control_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=control_pulse_time_series
                * self.MAX_AMP,
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

        all_timesteps_control_unitaries = compute_unitaries_for_all_time_steps(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_ctrl,
        )

        noise = self.noise_generator.generate_noise_instance(
            batch_size=batch_size
        )

        noise_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=noise,
                operators_for_time_evolving_elements=batched_noise_operators,
            )
        )

        interaction_hamiltonian = create_interaction_hamiltonian_for_each_timestep_noise_relisation_batchwise(
            control_unitaries=all_timesteps_control_unitaries,
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

        self.lastest_simulation_final_control_unitaries = (
            all_timesteps_control_unitaries[:, -1]
        )

        return (
            construct_vo_operator_for_batch(
                final_step_control_unitaries=self.lastest_simulation_final_control_unitaries,
                final_step_interaction_unitaries=final_timestep_interaction_unitaries,
            ),
            all_timesteps_control_unitaries[:, -1],
        )

    def compute_expectations(
        self,
        Vo_operators: torch.Tensor,
        final_control_unitaries: Optional[torch.Tensor] = None,
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

    def compute_fidelity_vo_operators_and_final_control_unitaries(
        self,
        vo_operators: torch.Tensor,
        final_control_unitaries: torch.Tensor,
    ) -> List[float]:
        return fidelity_batch_of_matrices(
            rho=torch.cat(
                (vo_operators, final_control_unitaries.unsqueeze(0)),
                dim=0,
            ),
            sigma=self.ideal_matrices,
        )

    def compute_fidelity_score(
        self,
        fidelities: torch.Tensor,
        num_qubits: int,
    ) -> List[float]:
        return (2 ** (2 * num_qubits) - torch.sum(fidelities, dim=0)).tolist()

    def time_series_to_score_wrapper(
        self,
        time_series: torch.Tensor,
    ) -> List[float]:
        (
            vo_operators,
            final_control_unitaries,
        ) = self.compute_vo_operators_and_final_control_unitaries(
            control_pulse_time_series=time_series,
        )

        return (
            compute_nuc_norm_of_diff_between_batch_of_matrices(
                rho=torch.cat(
                    (vo_operators, final_control_unitaries.unsqueeze(0)),
                    dim=0,
                ),
                sigma=self.ideal_matrices,
            )
            .sum(dim=0)
            .tolist()
        )


def generate_mixed_pulse(
    sequence_length,
    num_channels,
    random_noise_amount,
    cosine_amount,
    fund_freq,
    freq_range_max,
    num_pulses,
):
    t = torch.linspace(0, 1, sequence_length)
    freqs = fund_freq * torch.linspace(0, freq_range_max, num_pulses)

    random_signal = (
        2
        * torch.rand(
            (num_pulses, sequence_length, num_channels), device=DEVICE
        )
        - 1
    )

    sine_waves = (
        torch.sin(2 * torch.pi * freqs[:, None, None] * t[None, :, None])
        .to(DEVICE)
        .repeat(1, 1, num_channels)
    )

    cosine_waves = (
        torch.cos(2 * torch.pi * freqs[:, None, None] * t[None, :, None])
        .to(DEVICE)
        .repeat(1, 1, num_channels)
    )

    return (1 - random_noise_amount) * (
        (1 - cosine_amount) * sine_waves + cosine_amount * cosine_waves
    ) + random_noise_amount * random_signal


if NUM_QUBITS == 1:
    control_static_operators = 0.5 * QUBIT_ENERGY_GAP * SIGMA_Z

    control_dynamic_operators = torch.stack(
        (0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0
    )

    noise_dynamic_operators = torch.stack(
        (0.5 * SIGMA_X, 0.5 * SIGMA_Z), dim=0
    )

if NUM_QUBITS == 2:
    control_static_operators = torch.stack(
        (
            0.5 * QUBIT_ENERGY_GAP * torch.kron(SIGMA_Z, SIGMA_I),
            0.5 * QUBIT_ENERGY_GAP * torch.kron(SIGMA_I, SIGMA_Z),
        ),
        dim=0,
    )

    control_dynamic_operators = torch.stack(
        (
            0.5 * torch.kron(SIGMA_X, SIGMA_I),
            0.5 * torch.kron(SIGMA_Y, SIGMA_I),
            0.5 * torch.kron(SIGMA_I, SIGMA_X),
            0.5 * torch.kron(SIGMA_I, SIGMA_Y),
            0.5 * torch.kron(SIGMA_X, SIGMA_X),
        ),
        dim=0,
    )

    noise_dynamic_operators = torch.stack(
        (
            0.5 * torch.kron(SIGMA_X, SIGMA_I),
            0.5 * torch.kron(SIGMA_Z, SIGMA_I),
            0.5 * torch.kron(SIGMA_I, SIGMA_X),
            0.5 * torch.kron(SIGMA_I, SIGMA_Z),
        ),
        dim=0,
    )


identity_for_vo = (
    torch.eye(2**NUM_QUBITS)
    .repeat(2 ** (2 * NUM_QUBITS) - 1, 1, 1, 1)
    .to(DEVICE)
    .to(torch.cfloat)
)


def find_optimal_control_pulses(
    initial_number_of_pulses: int,
    num_optimisation_iterations: int,
    sequence_length: int,
    initial_pulses_random_noise_amount: float,
    initial_pulses_cosine_amount: float,
    initial_pulses_fund_freq: float,
    initial_pulses_freq_range_max: float,
    verbose: bool = False,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    optimal_pulse_sequences = {}

    bounds = np.array([[-1, 1]] * sequence_length * NUMBER_OF_CONTROL_CHANELS)

    n3_noise_generator = N3NoiseGenerator(
        number_of_noise_realisations=NUMBER_OF_NOISE_RELISATIONS,
        number_of_time_steps=sequence_length,
    )

    n6_noise_generator = SquaredScaledNoiseGenerator()

    n3n6_noise_generator = CombinedNoiseGenerator(
        x_noise_generator=n3_noise_generator,
        z_noise_generator=n6_noise_generator,
    )

    sim = QubitSimulator(
        delta_t=TOTAL_TIME / sequence_length,
        MAX_AMP=MAX_AMP,
        num_qubits=NUM_QUBITS,
        noise_dynamic_operators=noise_dynamic_operators,
        control_static_operators=control_static_operators,
        control_dynamic_operators=control_dynamic_operators,
        noise_generator=n3n6_noise_generator,
    )

    best_scores = []

    for gate in UNIVERSAL_GATE_SET_SINGLE_QUBIT.keys():
        inital_time_series = generate_mixed_pulse(
            sequence_length=sequence_length,
            num_channels=NUMBER_OF_CONTROL_CHANELS,
            random_noise_amount=initial_pulses_random_noise_amount,
            cosine_amount=initial_pulses_cosine_amount,
            fund_freq=initial_pulses_fund_freq,
            freq_range_max=initial_pulses_freq_range_max,
            num_pulses=initial_number_of_pulses,
        )

        kernel = ConstantKernel() * RBF() + ConstantKernel()
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

        ideal_matrices = torch.concatenate(
            [
                identity_for_vo,
                UNIVERSAL_GATE_SET_SINGLE_QUBIT[gate]
                .unsqueeze(0)
                .unsqueeze(0),
            ]
        )

        sim.change_ideal_matrices(new_ideal_matrices=ideal_matrices)

        with torch.no_grad():
            scores = sim.time_series_to_score_wrapper(
                time_series=inital_time_series
            )

            inital_time_series_flattened = (
                inital_time_series.view(inital_time_series.size(0), -1)
                .cpu()
                .numpy()
            )

            best_score = min(scores)

            best_sequence = inital_time_series[
                scores.index(best_score)
            ].unsqueeze(0)

            best_time_series_np = inital_time_series_flattened[
                scores.index(best_score)
            ]

            time_series_data_for_gp = inital_time_series_flattened
            scores_for_gp = scores

            gp.fit(time_series_data_for_gp, scores_for_gp)

            for iter in range(num_optimisation_iterations):
                res = minimize(
                    fun=lambda x, gp, best_score: -expected_improvement_for_minimising_obj_func(
                        x=x,
                        gp_model=gp,
                        y_min=best_score,
                    ),
                    x0=best_time_series_np,
                    bounds=bounds,
                    args=(gp, best_score),
                    options={"maxiter": 100},
                )

                new_time_series = (
                    torch.tensor(res.x)
                    .view(1, sequence_length, NUMBER_OF_CONTROL_CHANELS)
                    .to(DEVICE)
                    .to(torch.float32)
                )

                new_score = sim.time_series_to_score_wrapper(
                    time_series=new_time_series
                )[0]

                if verbose:
                    print(f"Iteration {iter+1}: Score = {round(new_score, 4)}")

                np_time_series = np.expand_dims(res.x, axis=0)

                time_series_data_for_gp = np.concatenate(
                    (time_series_data_for_gp, np_time_series), axis=0
                )

                scores_for_gp.append(new_score)

                gp.fit(time_series_data_for_gp, scores_for_gp)

                if new_score < best_score:
                    best_score = new_score
                    best_sequence = new_time_series
                    best_time_series_np = np_time_series

        best_score = sim.time_series_to_score_wrapper(
            time_series=best_sequence
        )[0]

        best_scores.append(best_score)

        optimal_pulse_sequences[gate] = best_sequence.squeeze()

        print(f"Gate = {gate}")
        print(f"Best Score = {best_score}")

    # you want to minimise the number of pulses and the number of
    # iterations, we also want to minimise the score (nuc norm), so
    # overall we want to mimimise the following
    return optimal_pulse_sequences, (
        sum(best_scores)
        + initial_number_of_pulses / MAX_INITIAL_PULSES
        + num_optimisation_iterations / MAX_ITERATIONS
    )
