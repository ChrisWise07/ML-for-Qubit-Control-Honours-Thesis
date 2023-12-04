from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, List
import torch
import random
from time_series_to_noise.monte_carlo_qubit_simulation import *

from time_series_to_noise.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    SIGMA_I,
    UNIVERSAL_GATE_SET_SINGLE_QUBIT,
    IDEAL_PROCESS_MATRICES,
)

from time_series_to_noise.utils import (
    calculate_expectation_values,
    fidelity_batch_of_matrices,
    calculate_state_from_observable_expectations,
    compute_process_matrix_for_single_qubit,
    compute_process_fidelity,
)

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

from control_pulse_optimisation.TruncatedNormal import TruncatedNormal

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

QUBIT_ENERGY_GAP = 12.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TIME = 1.0
MAX_AMP = 100
NUM_QUBITS = 1
NUMBER_OF_NOISE_RELISATIONS = 2000
NUMBER_OF_CONTROL_CHANELS = 2 if NUM_QUBITS == 1 else 5
MAX_INITIAL_PULSES = 50
number_of_control_channels = 2
ROOT_PATH = "."


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
    max_amp: float
    num_qubits: int
    control_sequence_length: int
    noise_dynamic_operators: torch.Tensor
    control_static_operators: torch.Tensor
    control_dynamic_operators: torch.Tensor
    noise_generator: CombinedNoiseGenerator
    ideal_matrices: Optional[torch.Tensor] = None

    def compute_final_control_unitary(
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

        control_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=control_pulse_time_series
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

        return compute_unitary_at_timestep(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_ctrl,
        )

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

        all_timesteps_control_unitaries = compute_unitaries_for_all_time_steps(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_ctrl,
        )

        noise = self.noise_generator.generate_noise_instance(
            batch_size=batch_size
        )[:, : self.control_sequence_length]

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

        return (
            construct_vo_operator_for_batch(
                final_step_control_unitaries=all_timesteps_control_unitaries[
                    :, -1
                ],
                final_step_interaction_unitaries=final_timestep_interaction_unitaries,
            ),
            all_timesteps_control_unitaries[:, -1],
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

    def time_series_to_process_fidelity_score(
        self,
        time_series: torch.Tensor,
        ideal_process_matrix: torch.Tensor,
    ) -> torch.Tensor:
        (
            vo_operators,
            final_control_unitaries,
        ) = self.compute_vo_operators_and_final_control_unitaries(
            control_pulse_time_series=time_series,
        )

        expectations = self.compute_expectations(
            Vo_operators=vo_operators,
            final_control_unitaries=final_control_unitaries,
        )

        expectations_plus = expectations[:, :3]
        expectations_minus = expectations[:, 9:12]
        expectations_zero = expectations[:, 12:15]
        expectations_one = expectations[:, 15:]

        reconstructed_ones = calculate_state_from_observable_expectations(
            expectation_values=expectations_one,
            observables=COMBINED_SIGMA_TENSOR.squeeze(),
            identity=SIGMA_I,
        )

        reconstructed_zeros = calculate_state_from_observable_expectations(
            expectation_values=expectations_zero,
            observables=COMBINED_SIGMA_TENSOR.squeeze(),
            identity=SIGMA_I,
        )

        reconstructed_plus = calculate_state_from_observable_expectations(
            expectation_values=expectations_plus,
            observables=COMBINED_SIGMA_TENSOR.squeeze(),
            identity=SIGMA_I,
        )

        reconstructed_minus = calculate_state_from_observable_expectations(
            expectation_values=expectations_minus,
            observables=COMBINED_SIGMA_TENSOR.squeeze(),
            identity=SIGMA_I,
        )

        process_matrix = compute_process_matrix_for_single_qubit(
            rho_zero=reconstructed_zeros,
            rho_one=reconstructed_ones,
            rho_plus=reconstructed_plus,
            rho_minus=reconstructed_minus,
        )

        batch_size = process_matrix.shape[0]

        return compute_process_fidelity(
            process_matrix_one=process_matrix,
            process_matrix_two=ideal_process_matrix.repeat(batch_size, 1, 1),
        )


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


def mutate_control_pulse_sequence(
    original_control_pulse_sequence: torch.Tensor,
    noise_amount: float = 0.1,
) -> torch.Tensor:
    noise = 2 * torch.rand_like(original_control_pulse_sequence) - 1

    return torch.clamp(
        original_control_pulse_sequence + noise_amount * noise,
        min=-1.0,
        max=1.0,
    )


def return_init_sol(
    sequence_length: int,
    noise_amount: float,
    gate: str,
    population_size: int,
):
    ideal_noiseless = (
        torch.load(
            f"./new_noiseless_ideal_pulses_seq_len_{sequence_length}.pt",
        )[gate].to(DEVICE)
    ).repeat(population_size, 1, 1)

    return mutate_control_pulse_sequence(
        original_control_pulse_sequence=ideal_noiseless,
        noise_amount=noise_amount,
    )


def find_optimal_control_pulses(
    sequence_length: int,
    noise_amount: float,
    population_size: int,
    num_generations: int,
    crossover_rate: float,
    elitism_rate: float,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    num_elite_population = int(elitism_rate * population_size)

    time_steps = 512

    n3_noise_generator = N3NoiseGenerator(
        number_of_noise_realisations=NUMBER_OF_NOISE_RELISATIONS,
        number_of_time_steps=time_steps,
        total_time=1.0,
        g=0.8,
    )

    n6_noise_generator = SquaredScaledNoiseGenerator(g=0.8)

    n3n6_noise_generator = CombinedNoiseGenerator(
        x_noise_generator=n3_noise_generator,
        z_noise_generator=n6_noise_generator,
    )

    sim = QubitSimulator(
        delta_t=1 / time_steps,
        max_amp=MAX_AMP,
        num_qubits=NUM_QUBITS,
        control_sequence_length=sequence_length,
        noise_dynamic_operators=noise_dynamic_operators,
        control_static_operators=control_static_operators,
        control_dynamic_operators=control_dynamic_operators,
        noise_generator=n3n6_noise_generator,
    )

    fidelity_scores = {}
    optimal_pulse_sequences = {}

    for gate in UNIVERSAL_GATE_SET_SINGLE_QUBIT.keys():
        new_population = return_init_sol(
            sequence_length=sequence_length,
            noise_amount=noise_amount,
            gate=gate,
            population_size=population_size,
        )

        ideal_expectations = sim.compute_expectations(
            Vo_operators=identity_for_vo,
            final_control_unitaries=UNIVERSAL_GATE_SET_SINGLE_QUBIT[
                gate
            ].unsqueeze(0),
        )

        best_score = float("inf")
        best_time_series = None

        with torch.no_grad():
            for generation in range(num_generations):
                (
                    actual_vo_operators,
                    actual_final_control_unitaries,
                ) = sim.compute_vo_operators_and_final_control_unitaries(
                    control_pulse_time_series=new_population,
                )

                actual_expectations = sim.compute_expectations(
                    Vo_operators=actual_vo_operators,
                    final_control_unitaries=actual_final_control_unitaries,
                )

                scores = (
                    torch.abs(actual_expectations - ideal_expectations)
                    .max(dim=-1)
                    .values
                )

                current_best_score = torch.min(scores).item()
                current_best_score_index = torch.argmin(scores).item()

                current_best_time_series = new_population[
                    current_best_score_index
                ]

                if current_best_score < best_score:
                    best_score = current_best_score
                    best_time_series = current_best_time_series

                if verbose:
                    print(
                        f"Generation {generation+1} | "
                        + f"Best Score For Generation: {round(current_best_score, 4)} | "
                        + f"Best Score Ever: {round(best_score, 4)}"
                    )

                old_population = new_population
                new_population = torch.empty_like(new_population)
                probabilities = torch.softmax(-scores, dim=0)
                old_population_sorted = old_population[scores.argsort()]

                new_population[:num_elite_population] = old_population_sorted[
                    :num_elite_population
                ]

                for idx in range(num_elite_population, population_size):
                    parent_indices = list(
                        torch.utils.data.WeightedRandomSampler(
                            weights=probabilities,
                            num_samples=2,
                            replacement=False,
                        )
                    )

                    parent1, parent2 = old_population[parent_indices]

                    if torch.rand(1).item() < crossover_rate and sequence_length > 1:
                        crossover_point = random.randint(
                            1, sequence_length - 1
                        )

                        child_sequence = torch.cat(
                            (
                                parent1[:crossover_point],
                                parent2[crossover_point:],
                            )
                        )
                    else:
                        child_sequence = (
                            parent1 if torch.rand(1).item() < 0.5 else parent2
                        )

                    child_sequence = mutate_control_pulse_sequence(
                        child_sequence, noise_amount
                    )

                    new_population[idx] = child_sequence

                new_population = torch.clamp(
                    new_population,
                    min=-1.0,
                    max=1.0,
                )

        process_fidelity_score = sim.time_series_to_process_fidelity_score(
            time_series=best_time_series.unsqueeze(0),
            ideal_process_matrix=IDEAL_PROCESS_MATRICES[gate].unsqueeze(0),
        ).item()

        fidelity_scores[gate] = process_fidelity_score
        optimal_pulse_sequences[gate] = best_time_series

    return fidelity_scores, optimal_pulse_sequences
