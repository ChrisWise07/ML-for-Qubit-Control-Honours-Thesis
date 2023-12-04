import torch
from typing import Optional, Union
import numpy as np
from typing import Tuple

from functools import reduce
import torch.nn.functional as F
import torch.fft
import scipy


from time_series_to_noise.constants import (
    DEVICE,
    COMBINED_SIGMA_TENSOR_TWO_QUBITS,
    COMBINED_SIGMA_TENSOR,
)

NOISE_DIM = 2


def construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
    time_evolving_elements: torch.Tensor,
    operators_for_time_evolving_elements: torch.Tensor,
    operators_for_static_elements: Optional[torch.Tensor] = None,
):
    """
    Constructs a tensor of Hamiltonians for each batch, and timestep
    witin the batch, and each noise realisation of that batch and
    timestep. Multiples in an efficient manner, see the code for
    details.

    Args:
        time_evolving_elements (torch.Tensor):
            the time evolving elements such as control fields or noise
            processes. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                number_of_dynamic_operators
            )
        operators_for_time_evolving_elements (torch.Tensor):
            the operators that are multiplied with the time evolving
            elements. Expected Shape:
            (
                batch_size,
                number_of_dynamic_operators,
                system_dimension,
                system_dimension
            )
        operators_for_static_elements (torch.Tensor):
            the operators that are multiplied with the static elements
            that are not time evolving such as the energy gap.
            Expected Shape:
            (
                batch_size,
                number_of_static_operators,
                system_dimension,
                system_dimension
            )
        operators_dim (int, optional):
            The dimensions that the operators are in. Defaults to -3.


    Returns:
        torch.Tensor:
            Resulting Hamiltonians of shape:
            (
                batch_size,
                num_timesteps,
                number_of_noise_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    dim_time_evolving_elements = time_evolving_elements.dim()
    expand_amount = dim_time_evolving_elements - 2
    operators_dim = dim_time_evolving_elements - 1
    time_evolving_elements_expanded = time_evolving_elements[..., None, None]

    operators_for_time_evolving_elements_new_shape = (
        operators_for_time_evolving_elements.shape[0:1]
        + (1,) * expand_amount
        + operators_for_time_evolving_elements.shape[1:]
    )

    operators_for_time_evolving_elements_expanded = (
        operators_for_time_evolving_elements.view(
            operators_for_time_evolving_elements_new_shape
        )
    )

    dynamic_part = torch.sum(
        time_evolving_elements_expanded
        * operators_for_time_evolving_elements_expanded,
        dim=operators_dim,
    )

    if operators_for_static_elements is None:
        return dynamic_part

    operators_for_static_elements_new_shape = (
        operators_for_static_elements.shape[0:1]
        + (1,) * expand_amount
        + operators_for_static_elements.shape[1:]
    )

    operators_for_static_elements_expanded = (
        operators_for_static_elements.view(
            operators_for_static_elements_new_shape
        )
    )

    static_part = torch.sum(
        operators_for_static_elements_expanded, dim=operators_dim
    )

    return dynamic_part + static_part


def return_exponentiated_scaled_hamiltonians(
    hamiltonians: torch.Tensor,
    delta_T: float,
) -> torch.Tensor:
    """
    Computes the exponential of the scaled Hamiltonian for the given
    batch of Hamiltonians. Hamiltonians are scaled by the time step
    delta_T. This is to be used for the unitary evolution operator
    Trotter-Suzuki decomposition.

    Args:
        hamiltonians (torch.Tensor):
            Hamiltonian for which to compute the evolution operator.
            Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
        delta_T (float):
            Time step for the evolution.

    Returns:
        torch.Tensor:
            Resulting exponential of the scaled Hamiltonian of shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    scaled_hamiltonian = hamiltonians * (-1j * delta_T)
    return torch.linalg.matrix_exp(scaled_hamiltonian)


def compute_unitary_at_timestep(
    exponential_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the unitary at a given timestep for the given batch of
    exponential Hamiltonians. Computes for the whole batch and for each
    noise realisation. This uses the Trotter-Suzuki decomposition. Note
    you should only pass in the exponential Hamiltonians actually
    needed for the time step, i.e. if you want to compute the unitary
    at time step t, then you should only pass in the exponential
    Hamiltonians for time steps 0 to t.

    Args:
        exponential_hamiltonians (torch.Tensor):
            Exponential Hamiltonian for which to compute the evolution
            operator. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting unitary evolution operators of shape:
            (
                batch_size,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """
    (
        _,
        num_time_steps,
        *_,
    ) = exponential_hamiltonians.shape

    if num_time_steps == 1:
        return exponential_hamiltonians[:, 0]

    if num_time_steps % 2 == 1:
        last_matrix = exponential_hamiltonians[:, -1:]
        exponential_hamiltonians = exponential_hamiltonians[:, :-1]
    else:
        last_matrix = None

    even_exponential_hamiltonians = exponential_hamiltonians[:, 0::2]
    odd_exponential_hamiltonians = exponential_hamiltonians[:, 1::2]

    product = torch.matmul(
        odd_exponential_hamiltonians, even_exponential_hamiltonians
    )

    if last_matrix is not None:
        product = torch.cat([product, last_matrix], dim=1)

    return compute_unitary_at_timestep(product)


def compute_unitaries_for_all_time_steps(
    exponential_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the unitaries for all time steps for the given batch of
    exponentiated Hamiltonians. Computes for the whole batch and for
    each noise realisation. This uses the Trotter-Suzuki decomposition.

    Args:
        exponential_hamiltonians (torch.Tensor):
            Exponential Hamiltonian for which to compute the evolution
            operator. Expected Shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting unitary evolution operators of shape:
            (
                batch_size,
                num_timesteps,
                number_of_realisations (optional),
                system_dimension,
                system_dimension
            )
    """

    result_unitaries = torch.empty_like(exponential_hamiltonians)

    result_unitaries[:, 0, ...] = exponential_hamiltonians[:, 0, ...]

    for time_step in range(1, exponential_hamiltonians.shape[1]):
        result_unitaries[:, time_step, ...] = (
            exponential_hamiltonians[:, time_step, ...]
            @ result_unitaries[:, time_step - 1, ...].clone()
        )

    return result_unitaries


def create_interaction_hamiltonian_for_each_timestep_noise_relisation_batchwise(
    control_unitaries: torch.Tensor,
    noise_hamiltonians: torch.Tensor,
) -> torch.Tensor:
    """
    Creates the interaction Hamiltonian for the given batch of control
    unitaries and noise Hamiltonians. Performs the calculation:
    H = U_ctrl^\dagger * H_noise * U_ctrl
    where U_dagger is the conjugate transpose of the control
    unitary and U_ctrl is the control unitary.

    Args:
        control_unitaries (torch.Tensor):
            Unitary evolution operator for the control part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                system_dimension,
                system_dimension
            )
        noise_hamiltonians (torch.Tensor):
            Hamiltonian for the noise part of the Hamiltonian.
            Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_realisations,
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting interaction Hamiltonian of shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_realisations,
                system_dimension,
                system_dimension
            )
    """
    control_unitaries_expanded = control_unitaries.unsqueeze(2)

    control_unitaries_expanded_dagger = torch.conj(
        control_unitaries_expanded
    ).transpose(-1, -2)

    return torch.matmul(
        control_unitaries_expanded_dagger,
        torch.matmul(noise_hamiltonians, control_unitaries_expanded),
    )


def __return_observables_for_vo_construction(
    system_dimension: int,
) -> torch.Tensor:
    """
    Returns the observables for the construction of the VO operator.
    Handles reshaping of the observables and if the signle or two qubit
    case is being considered.

    Args:
        system_dimension (int):
            The dimension of the system, i.e. the qubit

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            The observables for the construction of the VO operator.
            The first tensor is the observables constructing the
            ensemble average and so has an extra dimension for the
            number of realisations. The second tensor is the
            observables for the left multiplication of the ensemble
            average and so does not have that extra dimension. The shape
            of the first tensor is:
            (
                number_of_observables,
                1,
                1,
                system_dimension,
                system_dimension
            )
            The shape of the second tensor is:
            (
                number_of_observables,
                1,
                system_dimension,
                system_dimension
            )
    """
    if system_dimension == 2:
        return COMBINED_SIGMA_TENSOR.transpose(0, 1)

    return COMBINED_SIGMA_TENSOR_TWO_QUBITS.transpose(0, 1)


def construct_vo_operator_for_batch(
    final_step_control_unitaries: torch.Tensor,
    final_step_interaction_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Constructs the VO operator for the given batch of unitary
    interaction operators. Note the vo operator are stored in the
    order, Vx, Vy, Vz.

    Args:
        final_step_control_unitaries (torch.Tensor):
            Unitary evolution operator for the control part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                system_dimension,
                system_dimension
            )
        final_step_interaction_unitaries (torch.Tensor):
            Unitary evolution operator for the interaction part of the
            Hamiltonian. Expected Shape:
            (
                batch_size,
                number_of_realisations,
                system_dimension,
                system_dimension
            )

    Returns:
        torch.Tensor:
            Resulting VO operator of shape:
            (
                3,
                batch_size,
                system_dimension,
                system_dimension
            )
    """
    final_step_control_unitaries_expanded = (
        final_step_control_unitaries.unsqueeze(1)
    )

    final_step_control_unitaries_expanded_dagger = torch.conj(
        final_step_control_unitaries_expanded
    ).transpose(-1, -2)

    final_step_interaction_unitaries_tilde = torch.matmul(
        final_step_control_unitaries_expanded,
        torch.matmul(
            final_step_interaction_unitaries,
            final_step_control_unitaries_expanded_dagger,
        ),
    )

    final_step_interaction_unitaries_tilde_expanded = (
        final_step_interaction_unitaries_tilde[None, ...]
    )

    final_step_interaction_unitaries_tilde_dagger = torch.conj(
        final_step_interaction_unitaries_tilde_expanded
    ).transpose(-1, -2)

    observables = __return_observables_for_vo_construction(
        final_step_control_unitaries.shape[-1]
    )

    return (
        (observables @ final_step_interaction_unitaries_tilde_dagger)
        @ (observables @ final_step_interaction_unitaries_tilde_expanded)
    ).mean(dim=NOISE_DIM)


def generate_gaussian_pulses(
    number_of_channels: int,
    time_range_values: torch.Tensor,
    pulse_parameters: torch.Tensor,
) -> torch.Tensor:
    """
    Generates a gaussian pulse for the given parameters.

    Args:
        number_of_channels (int):
            Number of channels for which to generate the pulses.
        time_range_values (torch.Tensor):
            Time range for which to generate the pulses.
            Expected Shape:
            (
                number_of_time_steps,
            )
        pulse_parameters (torch.Tensor): Parameters of the pulses.
            Expected Shape:
            (
                batch_size,
                number_of_pulses,
                number_of_channels * 3
            )
            Where the last dimension contains the amplitude, position
            and standard deviation of the pulse.

    Returns:
        torch.Tensor:
            Resulting gaussian pulse of shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )

    """
    amplitude = torch.stack(
        [pulse_parameters[:, :, 0 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    position = torch.stack(
        [pulse_parameters[:, :, 1 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    std = torch.stack(
        [pulse_parameters[:, :, 2 + 3 * i] for i in range(number_of_channels)],
        dim=-1,
    )

    time_range_expanded = time_range_values[None, None, :, None]
    amplitude_expanded = amplitude[:, :, None, :]
    position_expanded = position[:, :, None, :]
    std_expanded = std[:, :, None, :]

    signal = amplitude_expanded * torch.exp(
        -0.5 * ((time_range_expanded - position_expanded) / std_expanded) ** 2
    )

    return signal.sum(dim=1)


def create_DFT_matrix_of_LTI_transfer_func_for_signal_distortion(
    total_time: Union[float, int],
    number_of_time_steps: int,
) -> torch.Tensor:
    """
    A function which creates a linear time invariant system for
    signal distortion. Returns a tensor of values which are the DFT of
    the transfer function of the system. These values can be used to
    distort a signal.

    Args:
        total_time (Union[float, int]):
            Total time for which the system is simulated.
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        batch_size (int):
            Number of systems to simulate.

    Returns:
        torch.Tensor:
            Tensor of values which can be used to create a transfer
            function for the system. Shape:
            (
                batch_size,
                1
                number_of_time_steps,
                number_of_time_steps
            )
    """
    numerator, denominator = scipy.signal.cheby1(
        4, 0.1, 2 * np.pi * 20, analog=True
    )
    numerator = torch.tensor(numerator, dtype=torch.cfloat, device=DEVICE)
    denominator = torch.tensor(denominator, dtype=torch.cfloat, device=DEVICE)

    frequency_vector = torch.reshape(
        torch.fft.fftfreq(number_of_time_steps).to(DEVICE)
        * number_of_time_steps
        / total_time,
        (1, number_of_time_steps),
    )

    dft_matrix = torch.tensor(
        scipy.linalg.dft(number_of_time_steps, "sqrtn"),
        dtype=torch.cfloat,
        device=DEVICE,
    )

    H_numerator = torch.cat(
        [
            (1j * 2 * np.pi * frequency_vector) ** s
            for s in range(len(numerator) - 1, -1, -1)
        ],
        axis=0,
    )

    H_denominator = torch.cat(
        [
            (1j * 2 * np.pi * frequency_vector) ** s
            for s in range(len(denominator) - 1, -1, -1)
        ],
        axis=0,
    )

    frequency_response_matrix = torch.diag(
        (numerator @ H_numerator) / (denominator @ H_denominator)
    )

    dft_of_transfer_func = torch.reshape(
        dft_matrix.conj().T @ frequency_response_matrix @ dft_matrix,
        (1, 1, number_of_time_steps, number_of_time_steps),
    )

    return dft_of_transfer_func


def generate_distorted_signal(
    original_signal: torch.Tensor,
    dft_matrix_of_transfer_func: torch.Tensor,
) -> torch.Tensor:
    """
    A function that generates a distorted signal from the given original signal.

    Args:
        original_signal (torch.Tensor):
            The original signal to distort. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )

    Returns:
        torch.Tensor:
            The distorted signal. Expected Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_channels
            )
    """
    (
        batch_size,
        number_of_time_steps,
        number_of_channels,
    ) = original_signal.shape

    x = original_signal.permute(0, 2, 1).to(dtype=torch.cfloat)

    x = torch.reshape(
        x, (batch_size, number_of_channels, number_of_time_steps, 1)
    )

    return torch.real(
        torch.matmul(dft_matrix_of_transfer_func, x).squeeze()
    ).permute(0, 2, 1)


def generate_spectal_density_noise_profile_one(
    frequencies: torch.Tensor,
    alpha: float = 1,
) -> torch.Tensor:
    """
    Generate a spectral density that has 1/f PSD with gaussian bump.
    See: https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        frequencies (torch.Tensor):
            The frequencies for which to generate the spectral density.
            Shape:
            (
                number_of_frequencies // 2,
            )
    """
    positive_frequencies = frequencies[frequencies >= 0]

    return (
        (1 / (positive_frequencies + 1) ** alpha)
        * (positive_frequencies <= 15)
        + (1 / 16) * (positive_frequencies > 15)
        + torch.exp(-((positive_frequencies - 30) ** 2) / 50) / 2
    )


def generate_spectal_density_noise_profile_five(
    frequencies: torch.Tensor,
    alpha: float = 1,
) -> torch.Tensor:
    """
    Generate a spectral density that has 1/f PSD. See:
    https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        frequencies (torch.Tensor):
            The frequencies for which to generate the spectral density.
            Shape:
            (
                number_of_frequencies // 2,
            )
    """
    return 1 / (frequencies[frequencies >= 0] + 1) ** alpha


def generate_sqrt_scaled_spectral_density(
    total_time: Union[float, int],
    spectral_density: torch.Tensor,
    number_of_time_steps: int,
) -> torch.Tensor:
    """
    A function which generates a spectral density that has been scaled
    and then square rooted. This spectral density can be used to
    generate noise with 1/f PSD.

    Args:
        total_time (Union[float, int]):
            Total time for which the system is simulated.
        spectral_density (torch.Tensor):
            The spectral density to scale and square root. Shape:
            (
                number_of_time_steps,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.

    Returns:
        torch.Tensor:
            Tensor of values which can be used to generate noise with
            1/f PSD. Shape:
            (
                number_of_time_steps // 2
            )
    """
    return torch.sqrt(
        spectral_density * (number_of_time_steps**2) / total_time
    )


#
def gerenate_noise_time_series_with_one_on_f_noise(
    sqrt_scaled_spectral_density: torch.Tensor,
    number_of_time_steps: int,
    number_of_noise_realisations: int,
    batch_size: int,
) -> torch.Tensor:
    """
    A function that generates a noise time domain signal.

    Args:
        sqrt_scaled_spectral_density (torch.Tensor):
            The spectral density to scale and square root. Shape:
            (
                number_of_time_steps // 2,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.
        batch_size (int): Number of systems to simulate.

    Returns:
        torch.Tensor:
            The generated noise signal. Shape:
            (
                batch_size,
                number_of_time_steps,
                number_of_noise_realisations,

            )
    """
    sqrt_scaled_spectral_density_expanded = sqrt_scaled_spectral_density[
        None, None, ...
    ]

    spectral_density_randomised_phases = (
        sqrt_scaled_spectral_density_expanded
        * torch.exp(
            2
            * torch.pi
            * 1j
            * torch.rand(
                (
                    batch_size,
                    number_of_noise_realisations,
                    number_of_time_steps // 2,
                ),
                device=DEVICE,
            )
        )
    )

    noise = torch.fft.ifft(
        torch.concatenate(
            (
                spectral_density_randomised_phases,
                spectral_density_randomised_phases.conj().flip(dims=[-1]),
            ),
            dim=-1,
        ),
        dim=-1,
    ).real

    return noise.permute(0, 2, 1)


def generate_time_domain_signal_for_noise(
    time_range: torch.Tensor,
    total_time: Union[float, int],
) -> torch.Tensor:
    """
    Generate a time domain signal for to be used in the generation of
    non-stationary noise. Note that the time_range should be on the
    device for best performance, i.e. then the time domain signal will
    also be on the device. See:
    https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        time_range (torch.Tensor):
            The time range for which to generate the signal. Shape:
            (
                number_of_time_steps,
            )
        total_time (Union[float, int]):
            Total time for which the system is simulated.
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.

    Returns:
        torch.Tensor:
            The time domain signal. Shape:
            (
                number_of_time_steps,
            )
    """
    return 1 - (torch.abs(time_range - 0.5 * total_time) * 2)


def generate_colour_filter_for_noise(
    number_of_time_steps: int,
    division_factor: int = 4,
) -> torch.Tensor:
    """
    Generate a colour filter for coloured noise. This is a filter that
    is used to generate coloured noise. See:
    https://www.nature.com/articles/s41597-022-01639-1
    for details.

    Args:
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        division_factor (int, optional):
            The division factor for the colour filter. Defaults to 4.
            This will control the frequency of the noise.

    Returns:
        torch.Tensor:
            The colour filter. Shape:
            (
                number_of_time_steps // division_factor,
            )
    """
    return torch.ones((number_of_time_steps // division_factor)).to(DEVICE)


def generate_non_stationary_colour_gaussian_noise_time_series(
    batch_size: int,
    time_domain_signal: torch.Tensor,
    colour_filter: torch.Tensor,
    number_of_time_steps: int,
    number_of_noise_realisations: int,
    division_factor: int = 4,
    g=0.2,
):
    """
    Generate non-stationary coloured Gaussian noise with specified
    time domain signal and colour filter. Can implement noise profile 3
    and 4 See: https://www.nature.com/articles/s41597-022-01639-1 for
    details.

    Args:
        batch_size (int):
            Number of systems to simulate.
        time_domain_signal (torch.Tensor):
            The time domain signal. Shape:
            (
                number_of_time_steps,
            )
        colour_filter (torch.Tensor):
            The colour filter. Shape:
            (
                number_of_time_steps // division_factor,
            )
        number_of_time_steps (int):
            Number of time steps for which the system is simulated.
        number_of_noise_realisations (int):
            Number of noise realisations to generate.
        division_factor (int, optional):
            The division factor for the colour filter. Defaults to 4.
            This will control the frequency of the noise.
        g (float, optional):
            Controls the strength of the noise. Defaults to 0.2.

    """
    random_numbers = torch.randn(
        (
            batch_size * number_of_noise_realisations,
            1,
            number_of_time_steps
            + (number_of_time_steps // division_factor)
            - 1,
        ),
        device=DEVICE,
    )

    colour_filter_expanded = colour_filter[None, None, ...]

    noise = g * F.conv1d(
        random_numbers, colour_filter_expanded, padding="valid"
    )

    noise = noise.view(batch_size, number_of_noise_realisations, -1)
    noise = noise.permute(0, 2, 1)
    return noise * time_domain_signal[None, :, None]


def square_and_scale_noise_time_series(
    noise_time_series: torch.Tensor,
    g: float = 0.3,
) -> torch.Tensor:
    """
    Square and scale the noise time series. See:
    https://www.nature.com/articles/s41597-022-01639-1 for details.

    Args:
        noise_time_series (torch.Tensor):
            The noise time series. Shape:
            (
                number_of_time_steps,
                number_of_noise_realisations,
            )
        g (float):
            The g value for the noise profile.

    Returns:
        torch.Tensor:
            The squared and scaled noise time series. Shape:
            (
                number_of_time_steps,
                number_of_noise_realisations,
            )
    """

    return g * torch.square(noise_time_series)
