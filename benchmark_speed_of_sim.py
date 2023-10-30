from dataclasses import dataclass
import torch

from time_series_to_noise.monte_carlo_qubit_simulation import *

from time_series_to_noise.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    SIGMA_I,
)

from time_series_to_noise.utils import (
    calculate_expectation_values,
)

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

        x_noise2 = self.x_noise_generator.generate_noise_instance(
            batch_size=batch_size
        )

        z_noise = self.z_noise_generator.generate_noise_instance(noise=x_noise)

        z_noise_2 = self.z_noise_generator.generate_noise_instance(
            noise=x_noise2
        )

        return torch.stack((x_noise, z_noise, x_noise2, z_noise_2), dim=-1)


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
    noise_generator: CombinedNoiseGenerator

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
        final_control_unitaries: Optional[torch.Tensor] = None,
        special_case: bool = False,
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
            special_case=special_case,
        )


# constant parameters
total_time = 1.0
max_amp = 100
num_qubits = 2
number_of_noise_realizations = 2000
number_of_control_channels = 2 if num_qubits == 1 else 3


sequence_length = 1024


if num_qubits == 1:
    control_static_operators = 0.5 * QUBIT_ENERGY_GAP * SIGMA_Z

    control_dynamic_operators = torch.stack(
        (0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0
    )

    noise_dynamic_operators = torch.stack(
        (0.5 * SIGMA_X, 0.5 * SIGMA_Z), dim=0
    )

if num_qubits == 2:
    control_static_operators = torch.stack(
        (
            0.5 * QUBIT_ENERGY_GAP * torch.kron(SIGMA_Z, SIGMA_I),
            0.5 * QUBIT_ENERGY_GAP * torch.kron(SIGMA_I, SIGMA_Z),
        ),
        dim=0,
    )

    control_dynamic_operators = torch.stack(
        (
            0.5 * torch.kron(SIGMA_I, SIGMA_X),
            0.5 * torch.kron(SIGMA_X, SIGMA_I),
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


n3_noise_generator = N3NoiseGenerator(
    number_of_noise_realisations=number_of_noise_realizations,
    number_of_time_steps=sequence_length,
)

n6_noise_generator = SquaredScaledNoiseGenerator()

n3n6_noise_generator = CombinedNoiseGenerator(
    x_noise_generator=n3_noise_generator,
    z_noise_generator=n6_noise_generator,
)


sim = QubitSimulator(
    delta_t=total_time / sequence_length,
    max_amp=max_amp,
    num_qubits=num_qubits,
    noise_dynamic_operators=noise_dynamic_operators,
    control_static_operators=control_static_operators,
    control_dynamic_operators=control_dynamic_operators,
    noise_generator=n3n6_noise_generator,
)


total_examples = 2000
batch_size = 10
num_iters = total_examples // batch_size


expectations_to_save = torch.empty(total_examples, 540)
vo_operators_to_save = torch.empty(15, total_examples, 4, 4, dtype=torch.cfloat)

control_pulse_time_series_to_save = torch.empty(
    total_examples, sequence_length, number_of_control_channels
)

control_unitaries_to_save = torch.empty(
    total_examples, 4, 4, dtype=torch.cfloat
)


for i in range(num_iters):
    with torch.no_grad():
        control_pulse_sequence = (
            2
            * torch.rand(
                (batch_size, sequence_length, number_of_control_channels),
                device=DEVICE,
            )
            - 1
        )

        (
            actual_vo_operators,
            actual_final_control_unitaries,
        ) = sim.compute_vo_operators_and_final_control_unitaries(
            control_pulse_time_series=control_pulse_sequence
        )

        actual_expectations = sim.compute_expectations(
            Vo_operators=actual_vo_operators,
            final_control_unitaries=actual_final_control_unitaries,
        )

        expectations_to_save[
            i * batch_size : (i + 1) * batch_size
        ] = actual_expectations.squeeze().cpu()

        vo_operators_to_save[
            :, i * batch_size : (i + 1) * batch_size
        ] = actual_vo_operators.squeeze().cpu()

        control_pulse_time_series_to_save[
            i * batch_size : (i + 1) * batch_size
        ] = control_pulse_sequence.squeeze().cpu()

        control_unitaries_to_save[
            i * batch_size : (i + 1) * batch_size
        ] = actual_final_control_unitaries.squeeze().cpu()

torch.save(
    expectations_to_save,
    "expectations_to_save.pt",
)

torch.save(
    vo_operators_to_save,
    "vo_operators_to_save.pt",
)

torch.save(
    control_pulse_time_series_to_save,
    "control_pulse_time_series_to_save.pt",
)

torch.save(
    control_unitaries_to_save,
    "control_unitaries_to_save.pt",
)
