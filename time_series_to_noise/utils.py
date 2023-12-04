import torch
import numpy as np
import multiprocessing
import os
from torch import Tensor
from typing import Tuple, Dict
import pickle
import zipfile
from typing import List

from time_series_to_noise.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    LIST_OF_PAULI_EIGENSTATES,
    LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS,
    DEVICE,
    COMBINED_SIGMA_TENSOR,
    COMBINED_SIGMA_TENSOR_TWO_QUBITS,
    LAMBDA_MATRIX_FOR_CHI_MATRIX,
    MIXED,
)
from natsort import natsorted

from typing import Callable

torch.manual_seed(0)
np.random.seed(0)


def batch_based_matrix_trace(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculates the trace of a batch of matrices.

    Args:
        matrix (Tensor): batch of matrices of shape (batch_size, 2, 2)

    Returns:
        Tensor: trace of the batch of matrices
    """
    return torch.diagonal(matrix, offset=0, dim1=-1, dim2=-2).sum(-1)


def compute_fidelity(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fidelity between two unitary matrices.

    Args:
        U (torch.Tensor): A unitary matrix.
        V (torch.Tensor): Another unitary matrix.

    Returns:
        float: The fidelity between U and V.
    """
    U_dagger = U.conj().transpose(-1, -2)

    trace = batch_based_matrix_trace(torch.matmul(U_dagger, V))

    fidelity = torch.abs(trace) ** 2

    return fidelity


def compute_nuc_norm_of_diff_between_batch_of_matrices(
    rho: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the nuclear norm of the difference between two batches of
    density matrices.

    Args:
        rho (np.ndarray): density matrix
        sigma (np.ndarray): density matrix

    Returns:
        Tensor: trace distance
    """
    return torch.linalg.matrix_norm(rho - sigma, ord="nuc")


def fidelity_batch_of_matrices(
    rho: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the fidelity between two batches of density matrices.

    Args:
        rho (torch.Tensor): density matrix
        sigma (torch.Tensor): density matrix

    Returns:
        Tensor: fidelity
    """
    rho_dagger = rho.conj().transpose(-1, -2)
    sigma_dagger = sigma.conj().transpose(-1, -2)

    return torch.square(
        torch.abs(
            batch_based_matrix_trace(torch.matmul(rho_dagger, sigma))
            / (
                torch.sqrt(
                    batch_based_matrix_trace(torch.matmul(rho_dagger, rho))
                    * batch_based_matrix_trace(
                        torch.matmul(sigma_dagger, sigma)
                    )
                )
                + 1e-6
            )
        )
    )


def compute_mean_distance_matrix_for_batch_of_VO_matrices(
    estimated_VX: torch.Tensor,
    estimated_VY: torch.Tensor,
    estimated_VZ: torch.Tensor,
    VX_true: torch.Tensor,
    VY_true: torch.Tensor,
    VZ_true: torch.Tensor,
    distance_measure: Callable = compute_nuc_norm_of_diff_between_batch_of_matrices,
) -> torch.Tensor:
    """
    Calculates the mean distance between the estimated and true VO
    matrices for a batch of density matrices. Can be customised to
    calculate other distance measures. By default, the nuclear norm
    of the difference between the matrices is calculated.

    Args:
        estimated_VX (Tensor): estimated VO matrix for the x axis
        estimated_VY (Tensor): estimated VO matrix for the y axis
        estimated_VZ (Tensor): estimated VO matrix for the z axis
        VX_true (Tensor): true VO matrix for the x axis
        VY_true (Tensor): true VO matrix for the y axis
        VZ_true (Tensor): true VO matrix for the z axis
        distance_measure (Callable, optional):
            distance measure to use. Defaults to
            compute_nuc_norm_of_diff_between_batch_of_matrices.

    Returns:
        Tensor: mean distance between the estimated and true VO matrices
    """
    return torch.cat(
        [
            distance_measure(estimated_VX, VX_true),
            distance_measure(estimated_VY, VY_true),
            distance_measure(estimated_VZ, VZ_true),
        ]
    ).mean()


def calculate_trig_expo_funcs_for_batch(
    psi: torch.Tensor,
    theta: torch.Tensor,
) -> Tensor:
    """
    Construct the estimated noise encoding matrix V_O from the
    parameters psi, theta, mu and the inverse of the pauli observable O.

    Args:
        psi (Tensor): parameter of shape (batch_size,)
        theta (Tensor): parameter of shape (batch_size,)

    Returns:
        Tensor:
            estimated noise encoding matrix QDQ^{\dagger} of shape (
                batch_size, 2, 2
            )
    """
    cos_2theta = torch.cos(2 * theta)
    sin_2theta = torch.sin(2 * theta)
    exp_2ipsi = torch.exp(2j * psi)
    exp_minus2ipsi = torch.exp(-2j * psi)

    return torch.stack(
        [
            cos_2theta,
            -exp_2ipsi * sin_2theta,
            -exp_minus2ipsi * sin_2theta,
            -cos_2theta,
        ],
        dim=-1,
    ).reshape(-1, 2, 2)


def calculate_QDQdagger_with_nine_params(
    batch_parameters: torch.Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Construct QDQ^{\dagger} for a batch of parameters
    (assuming 9 parameters). The function assumes the following order
    (x_psi, x_theta, x_mu, y_psi, y_theta, y_mu, z_psi, z_theta, z_mu).
    Return the matrices for the x, y and z axis, in that order.

    Args:
        batch_parameters (Tensor): batch of parameters

    Returns:
        Tensor: calculated QDQ^{\dagger} matrices in the order x, y, z
    """
    (
        x_psi,
        x_theta,
        x_mu,
        y_psi,
        y_theta,
        y_mu,
        z_psi,
        z_theta,
        z_mu,
    ) = batch_parameters.unbind(dim=1)

    return (
        calculate_trig_expo_funcs_for_batch(x_psi, x_theta)
        * x_mu[:, None, None],
        calculate_trig_expo_funcs_for_batch(y_psi, y_theta)
        * y_mu[:, None, None],
        calculate_trig_expo_funcs_for_batch(z_psi, z_theta)
        * z_mu[:, None, None],
    )


def calculate_QDQdagger_twelve_params(
    psi: torch.Tensor,
    theta: torch.Tensor,
    delta: torch.Tensor,
    mu: torch.Tensor,
) -> Tensor:
    """
    Construct the estimated QDQ^{\dagger} matrix from the parameters
    psi, theta, delta and mu.

    Args:
        psi (Tensor): parameter of shape (batch_size,)
        theta (Tensor): parameter of shape (batch_size,)
        delta (Tensor): parameter of shape (batch_size,)
        mu (Tensor): parameter of shape (batch_size,)

    Returns:
        Tensor:
            QDQ^{\dagger} of shape (batch_size, 2, 2)
    """
    batch_size = psi.shape[0]
    q = torch.zeros((batch_size, 2, 2), dtype=torch.cfloat, device=DEVICE)

    q[:, 0, 0] = torch.exp(1j * (delta + psi)) * torch.cos(theta)
    q[:, 0, 1] = torch.exp(-1j * (delta - psi)) * torch.sin(theta)
    q[:, 1, 0] = -torch.exp(1j * (delta - psi)) * torch.sin(theta)
    q[:, 1, 1] = torch.exp(-1j * (delta + psi)) * torch.cos(theta)

    mu = mu.to(torch.cfloat)
    d = torch.diag_embed(torch.stack([mu, -mu], dim=1), dim1=-2, dim2=-1)

    qdq_dagger = q @ d @ q.conj().transpose(-1, -2)

    return qdq_dagger


def calculate_QDQdagger_with_twelve_params(
    batch_parameters: torch.Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Construct QDQ^{\dagger} for a batch of parameters. The function
    assumes the following order (x_psi, x_theta, x_delta, x_mu, y_psi,
    y_theta, y_delta, y_mu, z_psi, z_theta, z_delta, z_mu). Return the
    matrices for the x, y and z axis, in that order.

    Args:
        batch_parameters (Tensor): batch of parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            calculated QDQ^{\dagger} matrices in the order x, y, z
    """

    (
        x_psi,
        x_theta,
        x_delta,
        x_mu,
        y_psi,
        y_theta,
        y_delta,
        y_mu,
        z_psi,
        z_theta,
        z_delta,
        z_mu,
    ) = batch_parameters.unbind(dim=1)

    return (
        calculate_QDQdagger_twelve_params(x_psi, x_theta, x_delta, x_mu),
        calculate_QDQdagger_twelve_params(y_psi, y_theta, y_delta, y_mu),
        calculate_QDQdagger_twelve_params(z_psi, z_theta, z_delta, z_mu),
    )


def return_estimated_VO_unital_for_batch(
    batch_parameters: torch.Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Construct the estimated noise encoding matrices V_O for a batch of
    parameters.

    Args:
        batch_parameters: batch of parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            estimated V_O matrices for the x, y and z axis, in that
            order
    """
    if batch_parameters.shape[1] == 9:
        (
            x_QDQdagger,
            y_QDQdagger,
            z_QDQdagger,
        ) = calculate_QDQdagger_with_nine_params(batch_parameters)
    elif batch_parameters.shape[1] == 12:
        (
            x_QDQdagger,
            y_QDQdagger,
            z_QDQdagger,
        ) = calculate_QDQdagger_with_twelve_params(batch_parameters)
    else:
        raise ValueError(
            "The number of parameters must be 9 or 12. "
            f"Instead, {batch_parameters.shape[1]} parameters were given."
        )

    return (
        SIGMA_X @ x_QDQdagger,
        SIGMA_Y @ y_QDQdagger,
        SIGMA_Z @ z_QDQdagger,
    )


def load_data_zip(
    filename_dataset_path_tuple: Tuple[str, str, List[str]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        filename: Name of the file to load.
        path_to_dataset: Path to the dataset zip file.

    Returns:
        pulses: Array of pulses.
        Vx: Array of X Vo operators.
        Vy: Array of Y Vo operators.
        Vz: Array of Z Vo operators.
    """
    filename, path_to_dataset, data_of_interest = filename_dataset_path_tuple
    try:
        with zipfile.ZipFile(f"{path_to_dataset}.zip", mode="r") as fzip:
            with fzip.open(filename, "r") as f:
                data = np.load(f, allow_pickle=True)
                return [data[key] for key in data_of_interest]
    except zipfile.BadZipFile:
        pass


def load_Vo_dataset_zip(
    path_to_dataset: str, num_examples: int, data_of_interest: List[str]
) -> List[torch.Tensor]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        path_to_dataset: Path to the dataset zip file.
        num_examples: Number of examples to load.

    Returns:
        pulses: Tensor of pulses.
        Vx: Tensor of X Vo operators.
        Vy: Tensor of Y Vo operators.
        Vz: Tensor of Z Vo operators.
    """
    with zipfile.ZipFile(f"{path_to_dataset}.zip", mode="r") as fzip:
        filenames = fzip.namelist()[:num_examples]

    with multiprocessing.Pool() as pool:
        func = pool.map_async(
            load_data_zip,
            [
                (filename, path_to_dataset, data_of_interest)
                for filename in filenames
            ],
        )
        data = zip(*func.get())

    return [torch.tensor(np.array(d)) for d in data]


def load_data_from_pickle(
    filename_dataset_path_tuple: Tuple[str, str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        filename: Name of the file to load.
        path_to_dataset: Path to the dataset zip file.

    Returns:
        Tuple of numpy arrays containing (in order):
            pulses: Array of pulses.
            VO: Array of Vo operators.
            expectations: Array of expectation values.
            control_unitaries: Array of control unitaries.
            pulse_parameters: Array of pulse parameters.

    """
    filename, path_to_dataset = filename_dataset_path_tuple
    with open(f"{path_to_dataset}/{filename}", "rb") as f:
        data = pickle.load(f)
    return (
        data["pulses"],
        data["Vo_operator"],
        data["expectations"],
        data["U0"],
        data["pulse_parameters"],
    )


def load_qdataset(
    path_to_dataset: str, num_examples: int, use_pulse_parameters: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Load the dataset of pulses, Vo operators, expectation values,
    control unitaries, and pulse parameters.

    Args:
        path_to_dataset: Path to the dataset
        num_examples: Number of examples to load.

    Returns:
        Dictionary containing:
            Vx: torch.Tensor of X Vo operators.
            Vy: torch.Tensor of Y Vo operators.
            Vz: torch.Tensor of Z Vo operators.
            expectations: torch.Tensor of expectation values.
            control_unitaries: torch.Tensor of control unitaries.
            if use_pulse_parameters:
                pulse_parameters: torch.Tensor of pulse parameters.
            else:
                pulses: torch.Tensor of pulses.
    """
    filenames = natsorted(os.listdir(path_to_dataset))[:num_examples]

    with multiprocessing.Pool() as pool:
        func = pool.map_async(
            load_data_from_pickle,
            [[filename, path_to_dataset] for filename in filenames],
        )
        pulses, Vo, expectations, control_unitaries, pulse_parameters = zip(
            *func.get()
        )

        Vo = np.array(Vo)
        expectations = np.array(expectations).squeeze()
        control_unitaries = np.array(control_unitaries).squeeze()

        dict_of_data = {
            "Vo": torch.from_numpy(Vo.squeeze()),
            "expectations": torch.from_numpy(expectations),
            "control_unitaries": torch.from_numpy(control_unitaries)[:, -1],
        }

        if use_pulse_parameters:
            pulse_parameters = np.array(pulse_parameters).squeeze()
            dict_of_data["pulse_parameters"] = torch.from_numpy(
                pulse_parameters
            )
            return dict_of_data

        pulses = np.array(pulses).squeeze()
        dict_of_data["pulses"] = torch.from_numpy(pulses)
        return dict_of_data


def load_data(
    filename_dataset_path_tuple: Tuple[str, str, List[str]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        filename: Name of the file to load.
        path_to_dataset: Path to the dataset zip file.

    Returns:
        pulses: Array of pulses.
        Vx: Array of X Vo operators.
        Vy: Array of Y Vo operators.
        Vz: Array of Z Vo operators.
    """
    filename, path_to_dataset, data_of_interest = filename_dataset_path_tuple
    with open(f"{path_to_dataset}/{filename}", "rb") as f:
        data = pickle.load(f)
        return [data[key] for key in data_of_interest]


def load_Vo_dataset(
    path_to_dataset: str,
    num_examples: int,
    data_of_interest: List[str],
    need_extended: bool = False,
) -> List[torch.Tensor]:
    """
    Load the dataset of pulses and Vo operators.

    Args:
        path_to_dataset: Path to the dataset zip file.
        num_examples: Number of examples to load.

    Returns:
        pulses: Tensor of pulses.
        Vx: Tensor of X Vo operators.
        Vy: Tensor of Y Vo operators.
        Vz: Tensor of Z Vo operators.
    """
    filenames = os.listdir(path_to_dataset)

    filenames_pkl_only = [
        filename for filename in filenames if filename.endswith(".pkl")
    ]

    filenames = filenames_pkl_only[:num_examples]

    with multiprocessing.Pool() as pool:
        func = pool.map(
            load_data,
            [
                (filename, path_to_dataset, data_of_interest)
                for filename in filenames
            ],
        )
        data = zip(*func)

    if not need_extended:
        return [torch.tensor(np.array(d)) for d in data]

    with multiprocessing.Pool() as pool:
        func = pool.map(
            load_data,
            [
                (filename, data_path_extended, ["expectations", "pulses"])
                for filename in filenames
            ],
        )
        data_extended = zip(*func)

    return [torch.tensor(np.array(d)) for d in data], [
        torch.tensor(np.array(d)) for d in data_extended
    ]


def calculate_psi_theta_mu(
    qdq_dagger: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the parameters psi, theta, mu from the QDQ^{\dagger}
    matrix.

    Args:
        matrix (Tensor):
            QDQ^{\dagger} matrix of shape (batch_size, 2, 2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: psi, theta, mu
    """

    mu = torch.abs(torch.real(torch.linalg.eigvals(qdq_dagger)))[:, 0]

    theta = 0.5 * torch.acos(torch.real(qdq_dagger[:, 0, 0]) / mu)
    theta = theta.nan_to_num(0)

    psi = 0.5 * torch.imag(
        torch.log(qdq_dagger[:, 0, 1] / (-mu * torch.sin(2 * theta)))
    )
    psi = psi.nan_to_num(0)

    return psi, theta, mu


def calculate_ground_turth_parameters(
    ground_truth_VX: torch.Tensor,
    ground_truth_VY: torch.Tensor,
    ground_truth_VZ: torch.Tensor,
    need_delta: bool = False,
):
    """
    Calculate the ground truth parameters psi, theta, mu from the noise
    encoding matrices.

    Args:
        ground_truth_VX (Tensor): noise encoding matrix V_X
        ground_truth_VY (Tensor): noise encoding matrix V_Y
        ground_truth_VZ (Tensor): noise encoding matrix V_Z

    Returns:
        Tensor: ground truth parameters
    """
    X_psi, X_theta, X_mu = calculate_psi_theta_mu(
        SIGMA_X @ ground_truth_VX.to(DEVICE)
    )
    Y_psi, Y_theta, Y_mu = calculate_psi_theta_mu(
        SIGMA_Y @ ground_truth_VY.to(DEVICE)
    )
    Z_psi, Z_theta, Z_mu = calculate_psi_theta_mu(
        SIGMA_Z @ ground_truth_VZ.to(DEVICE)
    )

    if need_delta:
        return torch.stack(
            (
                X_psi,
                X_theta,
                torch.zeros_like(X_mu),
                X_mu,
                Y_psi,
                Y_theta,
                torch.zeros_like(Y_mu),
                Y_mu,
                Z_psi,
                Z_theta,
                torch.zeros_like(Z_mu),
                Z_mu,
            ),
            dim=1,
        ).real

    return torch.stack(
        (X_psi, X_theta, X_mu, Y_psi, Y_theta, Y_mu, Z_psi, Z_theta, Z_mu),
        dim=1,
    ).real


def __return_qubit_initial_states_and_observables_tensor(
    system_dimension: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A hidden function that returns the tensor of qubit initial states
    and observables where the tensors returned depends on the system
    dimension.

    Args:
        system_dimension: dimension of the system

    Returns:
        inital states: tensor of initial states of shape: (
            1,
            2^system_dimension - 1,
            1,
            system_dimension,
            system_dimension
        )
    """
    if system_dimension == 2:
        return LIST_OF_PAULI_EIGENSTATES, COMBINED_SIGMA_TENSOR

    return (
        LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS,
        COMBINED_SIGMA_TENSOR_TWO_QUBITS,
    )


def compute_ctrl_unitary_init_state_obvs_product(
    control_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the matrix product of the control unitaries with the
    initial states and observables.

    Args:
        control_unitaries: tensor of shape
            (
                batch_size,
                system_dim,
                system_dim
            )

    Returns:
        Tensor of shape (batch_size, 3, 6, system_dim, system_dim)
    """
    batch_size = control_unitaries.shape[0]

    (
        initial_qubit_states,
        observables,
    ) = __return_qubit_initial_states_and_observables_tensor(
        control_unitaries.shape[-1]
    )

    # tensor of shape (batch_size, 1, 6, system_dim, system_dim)
    initial_rho_states = initial_qubit_states.repeat(batch_size, 1, 1, 1, 1)

    # tensor of shape (batch_size, 1, 1, system_dim, system_dim)
    control_unitaries_dagger = (
        control_unitaries.conj().transpose(-1, -2).unsqueeze(1).unsqueeze(1)
    )

    # tensor of shape (batch_size, 1, 1, system_dim, system_dim)
    control_unitaries = control_unitaries.unsqueeze(1).unsqueeze(1)

    # combined sigma tensor has shape (1, 3, 1, system_dim, system_dim)

    # tensor of shape (batch_size, 3, 6, system_dim, system_dim)
    return torch.matmul(
        control_unitaries,
        torch.matmul(
            initial_rho_states,
            torch.matmul(control_unitaries_dagger, observables),
        ),
    )


def calculate_expectation_values(
    Vo_operators: torch.Tensor,
    control_unitaries: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the expectation values of a set of Pauli operators.
    Vo_operators is a tensor of shape (num_operators, ...), representing
    different observables. Returns a tensor of shape
    (batch_size, num_operators * num_qubits).

    Args:
        Vo_operators:
            tensor of shape:
                (
                    num_operators,
                    batch_size,
                    system_dimension,
                    system_dimension
                )
        control_unitaries:
            tensor of shape:
                (
                    batch_size,
                    system_dimension,
                    system_dimension
                )

    Returns:
        Tensor of shape (batch_size, num_operators * 6 ^ num_qubits).
        It is import to note the form of the expectation tensor. It will
        be in the form,
        [x_rho_1, y_rho_1, z_rho_1, ..., x_rho_n, y_rho_n, z_rho_n]
        where x_rho_1 for example is the expectation value of state
        rho_1 with respect to the Pauli operator sigma_x.

    """
    control_based_evolution_matrices = (
        compute_ctrl_unitary_init_state_obvs_product(control_unitaries)
    )

    expectation_values_list = [
        torch.matmul(
            single_Vo.unsqueeze(1),
            control_based_evolution_matrices[:, i],
        )
        .diagonal(offset=0, dim1=-1, dim2=-2)
        .sum(-1)
        for i, single_Vo in enumerate(Vo_operators)
    ]

    return (
        torch.stack(expectation_values_list, dim=2)
        .view(control_unitaries.shape[0], -1)
        .real
    )


def calculate_xyz_coefficents(
    X_rho: torch.Tensor,
    Y_rho: torch.Tensor,
    Z_rho: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the coefficients of the X, Y and Z expectation values.

    Args:
        X_rho: tensor of shape (batch_size, 6, 2, 2)
        Y_rho: tensor of shape (batch_size, 6, 2, 2)
        Z_rho: tensor of shape (batch_size, 6, 2, 2)

    Returns:
        Tuple of tensors of shape (batch_size, 6, 3)
    """
    # tensors of shape (batch_size, 6, 3)
    X_coefficients = torch.stack(
        [
            2 * X_rho[:, :, 0, 0].real,
            -2 * X_rho[:, :, 0, 0].imag,
            2 * X_rho[:, :, 0, 1].real - 1,
        ],
        dim=-1,
    )

    Y_coefficients = torch.stack(
        [
            2 * Y_rho[:, :, 0, 0].real,
            -2 * Y_rho[:, :, 0, 0].imag,
            2 * -Y_rho[:, :, 0, 1].imag - 1,
        ],
        dim=-1,
    )

    Z_coefficients = torch.stack(
        [
            -2 * Z_rho[:, :, 1, 0].real,
            -2 * Z_rho[:, :, 1, 0].imag,
            2 * Z_rho[:, :, 0, 0].real - 1,
        ],
        dim=-1,
    )

    return X_coefficients, Y_coefficients, Z_coefficients


def calculate_vo_alpha_beta_gamma(
    X_coefficients: torch.Tensor,
    Y_coefficients: torch.Tensor,
    Z_coefficients: torch.Tensor,
    expectations: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Calculate the parameters alpha, beta, gamma, and delta for the
    respective noise encoding matrices, Vx, Vy, and Vz.

    Args:
        expectation_values:
            expectation values of the form
            [x_rho_1, y_rho_1, z_rho_1, ..., x_rho_6, y_rho_6, z_rho_6]
        control_unitaries: the control unitaries

    Returns:
        Tensor: alpha, beta, gamma, and delta
    """

    return [
        torch.matmul(
            torch.pinverse(coefficient),
            expectations[:, index::3].unsqueeze(2),
        )
        .squeeze(2)
        .real
        for index, coefficient in enumerate(
            [X_coefficients, Y_coefficients, Z_coefficients]
        )
    ]


def construct_vo_operators_from_alpha_beta_solutions(
    x_sols: torch.Tensor,
    y_sols: torch.Tensor,
    z_sols: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Vx = torch.stack(
        [
            x_sols[..., 0] + x_sols[..., 1] * 1j,
            -x_sols[..., 2],
            x_sols[..., 2],
            x_sols[..., 0] - x_sols[..., 1] * 1j,
        ],
        dim=-1,
    ).view(batch_size, 2, 2)

    Vy = torch.stack(
        [
            y_sols[..., 0] + y_sols[..., 1] * 1j,
            y_sols[..., 2] * 1j,
            y_sols[..., 2] * 1j,
            y_sols[..., 0] - y_sols[..., 1] * 1j,
        ],
        dim=-1,
    ).view(batch_size, 2, 2)

    Vz = torch.stack(
        [
            z_sols[..., 2],
            -z_sols[..., 0] + z_sols[..., 1] * 1j,
            z_sols[..., 0] + z_sols[..., 1] * 1j,
            z_sols[..., 2],
        ],
        dim=-1,
    ).view(batch_size, 2, 2)

    return Vx, Vy, Vz


def calculate_xyz_vo_from_expectation_values_wrapper(
    expectation_values: torch.Tensor, control_unitaries: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates Vx, Vy, Vz from expectation values and control unitaries.

    Args:
        expectation_values:
            expectation values of the form
            [x_rho_1, y_rho_1, z_rho_1, ..., x_rho_6, y_rho_6, z_rho_6]
        control_unitaries: control unitaries

    Returns:
        List of Vx, Vy, Vz

    """
    batch_size = expectation_values.shape[0]

    xyz_rho = compute_ctrl_unitary_init_state_obvs_product(control_unitaries)

    x_coefficients, y_coefficients, z_coefficients = calculate_xyz_coefficents(
        X_rho=xyz_rho[:, 0],
        Y_rho=xyz_rho[:, 1],
        Z_rho=xyz_rho[:, 2],
    )

    x_sols, y_sols, z_sols = calculate_vo_alpha_beta_gamma(
        X_coefficients=x_coefficients,
        Y_coefficients=y_coefficients,
        Z_coefficients=z_coefficients,
        expectations=expectation_values,
    )

    return construct_vo_operators_from_alpha_beta_solutions(
        x_sols=x_sols, y_sols=y_sols, z_sols=z_sols, batch_size=batch_size
    )


def compute_convariance_in_uncertainity_for_xyz_alpha_beta_gamma(
    control_unitaries: torch.Tensor,
    expectation_variances: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Calculate the expectation values of the X, Y, and Z Pauli operators.

    """
    xyz_rho = compute_ctrl_unitary_init_state_obvs_product(control_unitaries)

    xyz_coefficient_matrices = calculate_xyz_coefficents(
        X_rho=xyz_rho[:, 0],
        Y_rho=xyz_rho[:, 1],
        Z_rho=xyz_rho[:, 2],
    )

    xyz_coefficient_matrices_inverse = [
        torch.pinverse(coefficient_matrix)
        for coefficient_matrix in xyz_coefficient_matrices
    ]

    xyz_expectations_covariance_matrices = [
        torch.diag_embed(expectation_variances[:, index::3])
        for index in range(3)
    ]

    return [
        coefficient_matrix_inverse
        @ expectation_covariance_matrix
        @ coefficient_matrix_inverse.transpose(-2, -1)
        for coefficient_matrix_inverse, expectation_covariance_matrix in zip(
            xyz_coefficient_matrices_inverse,
            xyz_expectations_covariance_matrices,
        )
    ]


def apply_tikhonov_regularization_to_covariance_matrix(
    covariance_matrix: np.ndarray,
    regularization_strength: float = 1e-6,
) -> np.ndarray:
    """
    Apply Tikhonov regularization to a covariance matrix.

    Args:
        covariance_matrix: covariance matrix
        regularization_strength: regularization strength

    Returns:
        Regularized covariance matrix

    """
    return covariance_matrix + regularization_strength * np.eye(
        covariance_matrix.shape[1]
    )


def calculate_state_from_observable_expectations(
    expectation_values: torch.Tensor,
    observables: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the state, that is, peform state tomography, from the
    expectation values and observables.

    Args:
        expectation_values:
            expectation values of the form
            [
                x_rho_1, y_rho_1, z_rho_1,
                ...,
                x_rho_n, y_rho_n, z_rho_n
            ]
            where n is the batch size. Expected shape: (
                batch_size, num_observables
            )
        observables:
            observables of the form
            [sigma_x, sigma_y, sigma_z]
            expected shape: (
                num_observables,
                system_dimension,
                system_dimension
            )
        identity:
            identity matrix with shape:
            (
                batch_size,
                system_dimension,
                system_dimension
            )

    Returns:
        Tensor: state of the system with shape:
            (
                batch_size,
                system_dimension,
                system_dimension
            )
    """
    observables_expanded = observables.unsqueeze(0)
    identity_expanded = identity.unsqueeze(0)
    expectation_values_expanded = expectation_values.unsqueeze(-1).unsqueeze(
        -1
    )

    return (
        1
        / 2
        * (
            identity_expanded
            + torch.sum(
                observables_expanded * expectation_values_expanded,
                dim=1,
            )
        )
    )


def compute_process_matrix_for_single_qubit(
    rho_zero: torch.Tensor,
    rho_one: torch.Tensor,
    rho_plus: torch.Tensor,
    rho_minus: torch.Tensor,
) -> torch.Tensor:
    """
    Following pg. 393 Box 8.5 of Nielsen and Chuang, calculate the chi
    matrix, i.e. the process matrix, for a single qubit.

    Args:
        rho_zero:
            the density matrix of the |0><0| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_one:
            the density matrix of the |1><1| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_plus:
            the density matrix of the |+><+| after the quantum process.
            expected shape: (batch_size, 2, 2)
        rho_minus:
            the density matrix of the |-><-| after the quantum process.
            expected shape: (batch_size, 2, 2)

    Returns:
        Tensor: chi/process matrix of shape (batch_size, 4, 4)
    """
    batch_size = rho_zero.shape[0]
    rho_one_prime = rho_zero
    rho_four_prime = rho_one

    rho_two_prime = (
        rho_plus
        - 1j * rho_minus
        - (1 - 1j) * (rho_one_prime + rho_four_prime) / 2
    )

    rho_three_prime = (
        rho_plus
        + 1j * rho_minus
        - (1 + 1j) * (rho_one_prime + rho_four_prime) / 2
    )

    rho_for_chi = torch.zeros(
        (batch_size, 4, 4), dtype=torch.cfloat, device=DEVICE
    )

    rho_for_chi[:, 0:2, 0:2] = rho_one_prime
    rho_for_chi[:, 0:2, 2:4] = rho_two_prime
    rho_for_chi[:, 2:4, 0:2] = rho_three_prime
    rho_for_chi[:, 2:4, 2:4] = rho_four_prime

    return (
        LAMBDA_MATRIX_FOR_CHI_MATRIX
        @ rho_for_chi
        @ LAMBDA_MATRIX_FOR_CHI_MATRIX
    )


def compute_process_fidelity(
    process_matrix_one: torch.Tensor,
    process_matrix_two: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the process fidelity between two process matrices. Note
    this method assumes one of the process matrices comes from a unitary
    process.

    Args:
        process_matrix_one:
            process matrix one with shape:
                (
                    batch_size,
                    system_dim ^ 2,
                    system_dim ^ 2
                )
        process_matrix_two:
            process matrix two with shape:
                (
                    batch_size,
                    system_dim ^ 2,
                    system_dim ^ 2
                )

    Returns:
        Tensor: process fidelity with shape (batch_size,)
    """
    return batch_based_matrix_trace(
        torch.matmul(process_matrix_one, process_matrix_two)
    ).real
