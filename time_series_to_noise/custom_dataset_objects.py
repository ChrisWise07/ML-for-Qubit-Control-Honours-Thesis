import torch
from time_series_to_noise.utils import (
    load_qdataset,
    calculate_ground_turth_parameters,
    compute_convariance_in_uncertainity_for_xyz_alpha_beta_gamma,
    construct_vo_operators_from_alpha_beta_solutions,
    apply_tikhonov_regularization_to_covariance_matrix,
)
from enum import Enum, auto
from scipy.stats import truncnorm
import math
import cmath
import numpy as np
from time_series_to_noise.truncated_mvn import TruncatedMVN
from typing import Tuple
from time_series_to_noise.constants import DEVICE


class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


PULSES_MAX = 100

torch.random.manual_seed(0)
np.random.seed(0)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_dataset: str,
        num_examples: int,
        need_delta: bool,
        use_timeseries: bool = False,
    ):
        self.path_to_dataset = path_to_dataset
        self.num_examples = num_examples

        data = load_qdataset(
            path_to_dataset=self.path_to_dataset,
            num_examples=self.num_examples,
            use_pulse_parameters=not use_timeseries,
        )

        if use_timeseries:
            self.pulse_data = data["pulses"] / PULSES_MAX
        else:
            self.pulse_data = data["pulse_parameters"]
            self.pulse_data[0, :] = self.pulse_data[0, :] / PULSES_MAX
            self.pulse_data[3, :] = self.pulse_data[3, :] / PULSES_MAX

        self.Vx = data["Vo"][:, 0]
        self.Vy = data["Vo"][:, 1]
        self.Vz = data["Vo"][:, 2]
        self.expectations = data["expectations"]
        self.control_unitaries = data["control_unitaries"]

        self.parameters = calculate_ground_turth_parameters(
            ground_truth_VX=self.Vx,
            ground_truth_VY=self.Vy,
            ground_truth_VZ=self.Vz,
            need_delta=need_delta,
        )

        self.mode = Mode.TRAIN

        self.expectation_covariance_matrices = (
            self._get_expectation_covariance_matrix(
                self._get_numpy_expectations()
            )
        )

    def __len__(self):
        return self.num_examples

    def _get_numpy_expectations(self):
        return self.expectations.cpu().numpy()

    def _return_data_for_test(self, idx):
        return {
            "pulse_data": self.pulse_data[idx],
            "parameters": self.parameters[idx],
            "Vx": self.Vx[idx],
            "Vy": self.Vy[idx],
            "Vz": self.Vz[idx],
            "expectations": self.expectations[idx],
            "control_unitaries": self.control_unitaries[idx],
        }

    def _get_expectation_covariance_matrix(self, np_expectations: np.ndarray):
        abs_expectations = abs(np_expectations)
        self.lower_bounds = (-1 - np_expectations) / (0.05 * abs_expectations)
        self.upper_bounds = (1 - np_expectations) / (0.05 * abs_expectations)
        self.scales = 0.05 * abs_expectations
        self.expecation_variances = truncnorm.var(
            self.lower_bounds,
            self.upper_bounds,
            loc=np_expectations,
            scale=self.scales,
        )
        expectation_covariance_matrices = np.expand_dims(
            self.expecation_variances, axis=1
        ) * np.eye(18)

        return apply_tikhonov_regularization_to_covariance_matrix(
            expectation_covariance_matrices
        )

    def _construct_vo_truncated_mvn_distributions(self):
        gpu_contol_unitaries = self.control_unitaries.to(DEVICE)
        
        (
            x_covariance_matrices,
            y_covariance_matrices,
            z_covariance_matrices,
        ) = compute_convariance_in_uncertainity_for_xyz_alpha_beta_gamma(
            control_unitaries=gpu_contol_unitaries,
            expectation_variances=torch.tensor(
                self.expecation_variances, dtype=torch.float32, device=DEVICE
            ),
        )

        del gpu_contol_unitaries

        x_covariance_matrices_np = (
            apply_tikhonov_regularization_to_covariance_matrix(
                x_covariance_matrices.cpu().numpy()
            )
        )
        y_covariance_matrices_np = (
            apply_tikhonov_regularization_to_covariance_matrix(
                y_covariance_matrices.cpu().numpy()
            )
        )
        z_covariance_matrices_np = (
            apply_tikhonov_regularization_to_covariance_matrix(
                z_covariance_matrices.cpu().numpy()
            )
        )

        Vx_np = self.Vx.cpu().numpy()
        Vy_np = self.Vy.cpu().numpy()
        Vz_np = self.Vz.cpu().numpy()

        lower_bounds = np.array([-1, -1, -1])
        upper_bounds = np.array([1, 1, 1])

        self.vo_truncated_mvn_distributions = [
            [
                TruncatedMVN(
                    mu=np.array(
                        [
                            Vx_np[i, 0, 0].real,
                            Vx_np[i, 0, 0].imag,
                            Vx_np[i, 1, 0].real,
                        ]
                    ),
                    cov=x_covariance_matrices_np[i, ...],
                    lb=lower_bounds,
                    ub=upper_bounds,
                ),
                TruncatedMVN(
                    mu=np.array(
                        [
                            Vy_np[i, 0, 0].real,
                            Vy_np[i, 0, 0].imag,
                            Vy_np[i, 1, 0].imag,
                        ]
                    ),
                    cov=y_covariance_matrices_np[i, ...],
                    lb=lower_bounds,
                    ub=upper_bounds,
                ),
                TruncatedMVN(
                    mu=np.array(
                        [
                            Vz_np[i, 1, 0].real,
                            Vz_np[i, 1, 0].imag,
                            Vz_np[i, 0, 0].real,
                        ]
                    ),
                    cov=z_covariance_matrices_np[i, ...],
                    lb=lower_bounds,
                    ub=upper_bounds,
                ),
            ]
            for i in range(self.num_examples)
        ]

    def _return_noisy_xyz_alpha_beta_gamma(self, idx):
        return [
            torch.tensor(
                self.vo_truncated_mvn_distributions[idx][i]
                .sample(1)
                .squeeze(1),
                dtype=torch.float32,
                device=DEVICE,
            )
            for i in range(3)
        ]

    def set_mode(self, mode: Mode):
        self.mode = mode

    def get_number_of_parameters_being_predicted(self):
        return self.parameters.shape[1]


class ExpectationDataset(BaseDataset):
    def __init__(
        self,
        path_to_dataset: str,
        num_examples: int,
        need_delta: bool,
        use_timeseries: bool = False,
    ):
        super().__init__(
            path_to_dataset, num_examples, need_delta, use_timeseries
        )

        np_expectations = self._get_numpy_expectations()

        self.expectation_truncated_mvn_distributions = [
            TruncatedMVN(
                mu=np_expectations[i],
                cov=self.expectation_covariance_matrices[i],
                lb=np.zeros((18,)) - 1,
                ub=np.ones((18,)),
            )
            for i in range(self.num_examples)
        ]

    def __getitem__(self, idx):
        if self.mode == Mode.TEST:
            return self._return_data_for_test(idx)

        if self.mode == Mode.VAL:
            return (
                self.pulse_data[idx],
                self.expectations[idx],
                self.control_unitaries[idx],
            )

        noisy_expectations = torch.tensor(
            self.expectation_truncated_mvn_distributions[idx]
            .sample(1)
            .squeeze(1),
            dtype=torch.float32,
            device=DEVICE,
        )

        return (
            self.pulse_data[idx],
            noisy_expectations,
            self.control_unitaries[idx],
        )


class VoDataset(BaseDataset):
    def __init__(
        self,
        path_to_dataset: str,
        num_examples: int,
        need_delta: bool,
        use_timeseries: bool = False,
    ):
        super().__init__(
            path_to_dataset, num_examples, need_delta, use_timeseries
        )
        self._construct_vo_truncated_mvn_distributions()

    def _return_noisy_Vx_Vy_Vz(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Add noise to the Vo matrix while making sure that the matrix
        remains sensible. The noise is 5% of the range of elements.

        Args:
            Vo (torch.Tensor): Vo matrix

        Returns:
            torch.Tensor: noisy Vo matrix
        """
        x_sols, y_sols, z_sols = self._return_noisy_xyz_alpha_beta_gamma(idx)

        vx, vy, vz = construct_vo_operators_from_alpha_beta_solutions(
            x_sols=x_sols,
            y_sols=y_sols,
            z_sols=z_sols,
            batch_size=1,
        )

        return vx.squeeze(0), vy.squeeze(0), vz.squeeze(0)

    def __getitem__(self, idx):
        if self.mode == Mode.TEST:
            return self._return_data_for_test(idx)

        if self.mode == Mode.VAL:
            return (
                self.pulse_data[idx],
                self.Vx[idx],
                self.Vy[idx],
                self.Vz[idx],
            )

        noisy_Vx, noisy_Vy, noisy_Vz = self._return_noisy_Vx_Vy_Vz(idx)

        return (
            self.pulse_data[idx],
            noisy_Vx,
            noisy_Vy,
            noisy_Vz,
        )


class ParamsDataset(BaseDataset):
    def __init__(
        self,
        path_to_dataset: str,
        num_examples: int,
        need_delta: bool,
        use_timeseries: bool = False,
    ):
        super().__init__(
            path_to_dataset, num_examples, need_delta, use_timeseries
        )
        self.need_delta = need_delta
        self._construct_vo_truncated_mvn_distributions()

    def _noisy_params(self, idx: int) -> torch.Tensor:
        x_sols, y_sols, z_sols = self._return_noisy_xyz_alpha_beta_gamma(idx)

        x_mu, y_mu, z_mu = [
            math.sqrt(sols[0] ** 2 + sols[1] ** 2 + sols[2] ** 2)
            for sols in [x_sols, y_sols, z_sols]
        ]

        x_theta, y_theta, z_theta = [
            math.acos(gamma / mu) / 2
            for gamma, mu in zip(
                [x_sols[2], y_sols[2], z_sols[2]], [x_mu, y_mu, z_mu]
            )
        ]

        x_psi, y_psi, z_psi = [
            cmath.log(-v12 / (mu * math.sin(2 * theta))).imag / 2
            for v12, mu, theta in zip(
                [
                    x_sols[0] - x_sols[1] * 1j,
                    -y_sols[0] * 1j - y_sols[1],
                    -z_sols[0] + z_sols[1] * 1j,
                ],
                [x_mu, y_mu, z_mu],
                [x_theta, y_theta, z_theta],
            )
        ]

        for psi in [x_psi, y_psi, z_psi]:
            if psi == math.nan:
                psi = 0.0

        if self.need_delta:
            tensor_to_return = torch.tensor(
                [
                    x_psi,
                    x_theta,
                    0.0,
                    x_mu,
                    y_psi,
                    y_theta,
                    0.0,
                    y_mu,
                    z_psi,
                    z_theta,
                    0.0,
                    z_mu,
                ],
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            tensor_to_return = torch.tensor(
                [
                    x_psi,
                    x_theta,
                    x_mu,
                    y_psi,
                    y_theta,
                    y_mu,
                    z_psi,
                    z_theta,
                    z_mu,
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

        if torch.isnan(tensor_to_return).any():
            print(
                f"x_sols: {x_sols}"
                + f"\ny_sols: {y_sols}"
                + f"\nz_sols: {z_sols}"
                + f"\nparams: {tensor_to_return}"
            )

        return tensor_to_return

    def __getitem__(self, idx):
        if self.mode == Mode.TEST:
            return self._return_data_for_test(idx)

        if self.mode == Mode.VAL:
            return self.pulse_data[idx], self.parameters[idx]

        noisy_params = self._noisy_params(idx)

        return self.pulse_data[idx], noisy_params


class CombinedDataset(BaseDataset):
    def __init__(
        self,
        path_to_dataset: str,
        num_examples: int,
        need_delta: bool,
        use_timeseries: bool = False,
    ):
        super().__init__(
            path_to_dataset, num_examples, need_delta, use_timeseries
        )

        self.need_delta = need_delta

        np_expectations = self._get_numpy_expectations()

        self.expectation_truncated_mvn_distributions = [
            TruncatedMVN(
                mu=np_expectations[i],
                cov=self.expectation_covariance_matrices[i],
                lb=np.zeros((18,)) - 1,
                ub=np.ones((18,)),
            )
            for i in range(self.num_examples)
        ]

        self._construct_vo_truncated_mvn_distributions()

    def _noisy_params(self, idx: int) -> torch.Tensor:
        x_sols, y_sols, z_sols = self._return_noisy_xyz_alpha_beta_gamma(idx)

        vx, vy, vz = construct_vo_operators_from_alpha_beta_solutions(
            x_sols=x_sols,
            y_sols=y_sols,
            z_sols=z_sols,
            batch_size=1,
        )

        x_mu, y_mu, z_mu = [
            math.sqrt(sols[0] ** 2 + sols[1] ** 2 + sols[2] ** 2)
            for sols in [x_sols, y_sols, z_sols]
        ]

        x_theta, y_theta, z_theta = [
            math.acos(gamma / mu) / 2
            for gamma, mu in zip(
                [x_sols[2], y_sols[2], z_sols[2]], [x_mu, y_mu, z_mu]
            )
        ]

        x_psi, y_psi, z_psi = [
            cmath.log(-v12 / (mu * math.sin(2 * theta))).imag / 2
            for v12, mu, theta in zip(
                [
                    x_sols[0] - x_sols[1] * 1j,
                    -y_sols[0] * 1j - y_sols[1],
                    -z_sols[0] + z_sols[1] * 1j,
                ],
                [x_mu, y_mu, z_mu],
                [x_theta, y_theta, z_theta],
            )
        ]

        x_psi = 0.0 if math.isnan(x_psi) else x_psi
        y_psi = 0.0 if math.isnan(y_psi) else y_psi
        z_psi = 0.0 if math.isnan(z_psi) else z_psi

        if self.need_delta:
            tensor_to_return = torch.tensor(
                [
                    x_psi,
                    x_theta,
                    0.0,
                    x_mu,
                    y_psi,
                    y_theta,
                    0.0,
                    y_mu,
                    z_psi,
                    z_theta,
                    0.0,
                    z_mu,
                ],
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            tensor_to_return = torch.tensor(
                [
                    x_psi,
                    x_theta,
                    x_mu,
                    y_psi,
                    y_theta,
                    y_mu,
                    z_psi,
                    z_theta,
                    z_mu,
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

        if torch.isnan(tensor_to_return).any():
            print(
                f"x_sols: {x_sols}"
                + f"\ny_sols: {y_sols}"
                + f"\nz_sols: {z_sols}"
                + f"\nparams: {tensor_to_return}"
            )

        return tensor_to_return, vx.squeeze(0), vy.squeeze(0), vz.squeeze(0)

    def __getitem__(self, idx):
        if self.mode == Mode.TEST:
            return self._return_data_for_test(idx)

        if self.mode == Mode.VAL:
            return (
                self.pulse_data[idx],
                self.parameters[idx],
                self.Vx[idx],
                self.Vy[idx],
                self.Vz[idx],
                self.expectations[idx],
                self.control_unitaries[idx],
            )

        noisy_params = self._noisy_params(idx)

        parameters = noisy_params[0]
        vx = noisy_params[1]
        vy = noisy_params[2]
        vz = noisy_params[3]

        noisy_expectations = torch.tensor(
            self.expectation_truncated_mvn_distributions[idx]
            .sample(1)
            .squeeze(1),
            dtype=torch.float32,
            device=DEVICE,
        )

        return [
            self.pulse_data[idx],
            parameters,
            vx,
            vy,
            vz,
            noisy_expectations,
            self.control_unitaries[idx],
        ]
