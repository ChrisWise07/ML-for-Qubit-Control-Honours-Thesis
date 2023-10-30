import unittest
import torch
from time_series_to_noise.monte_carlo_qubit_simulation import *
from time_series_to_noise.utils import calculate_expectation_values

import os
import multiprocessing
import numpy as np
import pickle
import zipfile
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

torch.set_printoptions(precision=6)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat).to(DEVICE)
SIGMA_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat).to(DEVICE)
SIGMA_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat).to(DEVICE)
SIGMA_I = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat).to(DEVICE)

NUM_EXAMPLES_FOR_TEST = 10

qubit_energy_gap = 12

root_dir = "/home/chriswise/github/Honours-Research-ML-for-QC/test"
data_path = f"{root_dir}/G_1q_XY_XZ_N1N5_D_reduced"
data_path_extended = "/home/chriswise/github/Honours-Research-ML-for-QC/QData_pickle/G_1q_XY_XZ_N1N5_comb/"
data_path2 = f"{root_dir}/G_1q_XY_XZ_N3N6_D_reduced"
figure_path = f"{root_dir}/graphs_to_verify_noise"


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
    with zipfile.ZipFile(f"{path_to_dataset}.zip", mode="r") as fzip:
        with fzip.open(filename, "r") as f:
            data = np.load(f, allow_pickle=True)
            return [data[key] for key in data_of_interest]


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

    load_data_zip((filenames[0], path_to_dataset, data_of_interest))


class TestHamiltonianConstruction(unittest.TestCase):
    def test_hamiltonian_construction_control_operators(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        num_timesteps = 1024
        system_dimension = 2

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["H0", "distorted_pulses"],
        )

        gt_control_hamiltonian = data[0].squeeze().to(DEVICE).to(torch.cfloat)
        pulses = data[1].squeeze().to(DEVICE)
        static_operators = 0.5 * qubit_energy_gap * SIGMA_Z
        batched_static_operators = static_operators.repeat(batch_size, 1, 1, 1)
        control_operators = torch.stack((0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0)

        batched_control_operators = control_operators.repeat(
            batch_size, 1, 1, 1
        )

        control_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=pulses,
                operators_for_time_evolving_elements=batched_control_operators,
                operators_for_static_elements=batched_static_operators,
            )
        )

        self.assertEqual(
            control_hamiltonian.shape,
            (
                batch_size,
                num_timesteps,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(control_hamiltonian.device.type, DEVICE.type)

        self.assertTrue(
            torch.allclose(control_hamiltonian, gt_control_hamiltonian)
        )

    def test_hamiltonian_construction_noise_operators(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        num_timesteps = 1024
        number_of_realisations = 2000
        system_dimension = 2

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["H1", "noise"],
        )

        gt_noise_hamiltonian = data[0].squeeze().to(DEVICE).to(torch.cfloat)
        noise = data[1].squeeze().to(DEVICE)
        noise_operators = torch.stack((0.5 * SIGMA_X, 0.5 * SIGMA_Z), dim=0)
        batched_noise_operators = noise_operators.repeat(batch_size, 1, 1, 1)

        noise_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=noise,
                operators_for_time_evolving_elements=batched_noise_operators,
            )
        )

        self.assertEqual(
            noise_hamiltonian.shape,
            (
                batch_size,
                num_timesteps,
                number_of_realisations,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(noise_hamiltonian.device.type, DEVICE.type)

        self.assertTrue(
            torch.allclose(noise_hamiltonian, gt_noise_hamiltonian)
        )


class TestUnitaryConstruction(unittest.TestCase):
    def test_unitary_construction_control_operators(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        system_dimension = 2
        num_timesteps = 1024

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["U0", "H0"],
        )

        gt_control_unitary = data[0].squeeze().to(DEVICE).to(torch.cfloat)

        hamiltonians = data[1].squeeze().to(DEVICE).to(torch.cfloat)
        num_timesteps = hamiltonians.shape[1]

        exponentiated_scaled_hamiltonians = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=hamiltonians,
                delta_T=1.0 / num_timesteps,
            )
        )

        self.assertEqual(
            exponentiated_scaled_hamiltonians.shape,
            (
                batch_size,
                num_timesteps,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(
            exponentiated_scaled_hamiltonians.device.type, DEVICE.type
        )

        all_timesteps_control_unitaries = compute_unitaries_for_all_time_steps(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians,
        )

        self.assertEqual(
            all_timesteps_control_unitaries.shape,
            (
                batch_size,
                num_timesteps,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(
            all_timesteps_control_unitaries.device.type, DEVICE.type
        )

        self.assertTrue(
            torch.allclose(
                all_timesteps_control_unitaries, gt_control_unitary, atol=1e-4
            )
        )

    def test_unitary_construction_noise_operators(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        number_of_realisations = 2000
        num_timesteps = 1024
        system_dimension = 2

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["UI", "H1", "U0"],
        )

        gt_noise_unitary = data[0].squeeze().to(DEVICE).to(torch.cfloat)
        hamiltonians_noise = data[1].squeeze().to(DEVICE).to(torch.cfloat)
        num_timesteps = hamiltonians_noise.shape[1]

        control_unitary = data[2].squeeze().to(DEVICE).to(torch.cfloat)

        interaction_hamiltonian = create_interaction_hamiltonian_for_each_timestep_noise_relisation_batchwise(
            control_unitaries=control_unitary,
            noise_hamiltonians=hamiltonians_noise,
        )

        self.assertEqual(
            interaction_hamiltonian.shape,
            (
                batch_size,
                num_timesteps,
                number_of_realisations,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(interaction_hamiltonian.device.type, DEVICE.type)

        exponentiated_scaled_hamiltonians = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=interaction_hamiltonian,
                delta_T=1.0 / num_timesteps,
            )
        )

        self.assertEqual(
            exponentiated_scaled_hamiltonians.shape,
            (
                batch_size,
                num_timesteps,
                number_of_realisations,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(
            exponentiated_scaled_hamiltonians.device.type, DEVICE.type
        )

        final_timestep_interaction_unitaries = compute_unitary_at_timestep(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians,
        )

        self.assertEqual(
            final_timestep_interaction_unitaries.shape,
            (
                batch_size,
                number_of_realisations,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(
            final_timestep_interaction_unitaries.device.type, DEVICE.type
        )

        diff = torch.abs(final_timestep_interaction_unitaries - gt_noise_unitary)

        self.assertTrue(
            torch.allclose(
                final_timestep_interaction_unitaries,
                gt_noise_unitary,
                atol=1e-4,
            )
        )


class TestVoOperatorConstruction(unittest.TestCase):
    def test_vo_operator(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        system_dimension = 2

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["Vo_operator", "UI", "U0"],
        )

        gt_vo_operators = data[0]

        gt_vx_operator = (
            gt_vo_operators[:, 0, :, :].squeeze().to(DEVICE).to(torch.cfloat)
        )

        gt_vy_operator = (
            gt_vo_operators[:, 1, :, :].squeeze().to(DEVICE).to(torch.cfloat)
        )

        gt_vz_operator = (
            gt_vo_operators[:, 2, :, :].squeeze().to(DEVICE).to(torch.cfloat)
        )

        interaction_unitary = data[1].squeeze().to(DEVICE).to(torch.cfloat)

        control_unitary = (
            data[2].squeeze()[:, -1, :, :].to(DEVICE).to(torch.cfloat)
        )

        vo_operator = construct_vo_operator_for_batch(
            final_step_control_unitaries=control_unitary,
            final_step_interaction_unitaries=interaction_unitary,
        )

        self.assertEqual(
            vo_operator.shape,
            (
                3,
                batch_size,
                system_dimension,
                system_dimension,
            ),
        )

        self.assertEqual(vo_operator.device.type, DEVICE.type)

        self.assertTrue(torch.allclose(vo_operator[0], gt_vx_operator))
        self.assertTrue(torch.allclose(vo_operator[1], gt_vy_operator))
        self.assertTrue(torch.allclose(vo_operator[2], gt_vz_operator))


class TestPulsesToExpectationValues(unittest.TestCase):
    def test_pulses_to_expectations(self):
        batch_size = NUM_EXAMPLES_FOR_TEST
        num_timesteps = 1024

        data, data2 = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=[
                "distorted_pulses",
                "noise",
                "pulses",
            ],
            need_extended=True,
        )

        gt_expectations = data2[0].squeeze().to(DEVICE)
        pulses = data[0].squeeze().to(DEVICE)
        noise = data[1].squeeze().to(DEVICE)
        control_static_operators = 0.5 * qubit_energy_gap * SIGMA_Z

        batched_static_operators = control_static_operators.repeat(
            batch_size, 1, 1, 1
        )

        control_dynamic_operators = torch.stack(
            (0.5 * SIGMA_X, 0.5 * SIGMA_Y), dim=0
        )

        batched_control_operators = control_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        noise_dynamic_operators = torch.stack(
            (0.5 * SIGMA_X, 0.5 * SIGMA_Z), dim=0
        )

        batched_noise_operators = noise_dynamic_operators.repeat(
            batch_size, 1, 1, 1
        )

        control_hamiltonian = (
            construct_hamiltonian_for_each_timestep_noise_relisation_batchwise(
                time_evolving_elements=pulses,
                operators_for_time_evolving_elements=batched_control_operators,
                operators_for_static_elements=batched_static_operators,
            )
        )

        exponentiated_scaled_hamiltonians_ctrl = (
            return_exponentiated_scaled_hamiltonians(
                hamiltonians=control_hamiltonian,
                delta_T=1.0 / num_timesteps,
            )
        )

        all_timesteps_control_unitaries = compute_unitaries_for_all_time_steps(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_ctrl,
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
                delta_T=1.0 / num_timesteps,
            )
        )

        final_timestep_interaction_unitaries = compute_unitary_at_timestep(
            exponential_hamiltonians=exponentiated_scaled_hamiltonians_interaction,
        )

        vo_operators = construct_vo_operator_for_batch(
            final_step_control_unitaries=all_timesteps_control_unitaries[
                :, -1
            ],
            final_step_interaction_unitaries=final_timestep_interaction_unitaries,
        )

        expectations = calculate_expectation_values(
            Vo_operators=vo_operators,
            control_unitaries=all_timesteps_control_unitaries[:, -1],
        )

        self.assertEqual(
            expectations.shape,
            (batch_size, 18),
        )

        self.assertEqual(expectations.device.type, DEVICE.type)
        diff = torch.abs(expectations - gt_expectations)
        self.assertTrue((diff.max() < 1e-3).item())
        self.assertTrue((diff.mean() < 1e-4).item())


class TestSignalGeneration(unittest.TestCase):
    def test_gaussian_pulse_construction(self):
        total_time = 1
        number_of_time_steps = 1024

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["pulse_parameters", "pulses"],
        )

        pulse_parameters = data[0].squeeze(1).to(DEVICE)
        gt_time_dom_rep_of_pulses = data[1].squeeze().to(DEVICE)

        step = 0.5 * total_time / number_of_time_steps

        time_range = torch.linspace(
            step,
            total_time - step,
            number_of_time_steps,
        ).to(DEVICE)

        pulses = generate_gaussian_pulses(
            number_of_channels=2,
            time_range_values=time_range,
            pulse_parameters=pulse_parameters,
        )

        self.assertEqual(
            pulses.shape,
            (
                NUM_EXAMPLES_FOR_TEST,
                1024,
                2,
            ),
        )

        self.assertEqual(pulses.device.type, DEVICE.type)

        self.assertTrue(torch.allclose(pulses, gt_time_dom_rep_of_pulses))

    def test_gaussian_pulse_construction_with_distortion(self):
        total_time = 1
        number_of_time_steps = 1024
        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["pulses", "distorted_pulses"],
        )

        pulses = data[0].squeeze().to(DEVICE)
        gt_distorted_pulses = data[1].squeeze().to(DEVICE)

        dft_matrix_of_transfer_func = (
            create_DFT_matrix_of_LTI_transfer_func_for_signal_distortion(
                total_time=total_time,
                number_of_time_steps=number_of_time_steps,
                batch_size=NUM_EXAMPLES_FOR_TEST,
            )
        )

        self.assertEqual(
            dft_matrix_of_transfer_func.shape,
            (
                NUM_EXAMPLES_FOR_TEST,
                1,
                1024,
                1024,
            ),
        )

        self.assertEqual(dft_matrix_of_transfer_func.device.type, DEVICE.type)

        distorted_pulses = generate_distorted_signal(
            original_signal=pulses,
            dft_matrix_of_transfer_func=dft_matrix_of_transfer_func,
        )

        self.assertEqual(
            distorted_pulses.shape,
            (
                NUM_EXAMPLES_FOR_TEST,
                1024,
                2,
            ),
        )

        self.assertEqual(distorted_pulses.device.type, DEVICE.type)

        self.assertTrue(
            torch.allclose(distorted_pulses, gt_distorted_pulses, atol=1e-4)
        )


class TestNoiseGeneration(unittest.TestCase):
    def test_noise_profile_1(self):
        plot_psd = False
        total_time = 1
        number_of_time_steps = 1024
        number_of_noise_realisations = 2000
        batch_size = NUM_EXAMPLES_FOR_TEST

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["noise"],
        )

        gt_noise = data[0].squeeze().to(DEVICE)
        n1_noise = gt_noise[:, :, :, 0]

        frequencies = torch.fft.fftfreq(
            n=number_of_time_steps, d=total_time / number_of_time_steps
        ).to(DEVICE)

        n1_spectral_density = generate_spectal_density_noise_profile_one(
            frequencies=frequencies,
            alpha=1,
        )

        self.assertEqual(
            n1_spectral_density.shape,
            (number_of_time_steps // 2,),
        )

        self.assertEqual(n1_spectral_density.device.type, DEVICE.type)

        n1_sqrt_scaled_spectral_density = (
            generate_sqrt_scaled_spectral_density(
                total_time=total_time,
                spectral_density=n1_spectral_density,
                number_of_time_steps=number_of_time_steps,
            )
        )

        self.assertEqual(
            n1_sqrt_scaled_spectral_density.shape,
            (number_of_time_steps // 2,),
        )

        self.assertEqual(
            n1_sqrt_scaled_spectral_density.device.type, DEVICE.type
        )

        noise = gerenate_noise_time_series_with_one_on_f_noise(
            sqrt_scaled_spectral_density=n1_sqrt_scaled_spectral_density,
            number_of_noise_realisations=number_of_noise_realisations,
            number_of_time_steps=number_of_time_steps,
            batch_size=batch_size,
        )

        self.assertEqual(
            noise.shape,
            (
                batch_size,
                number_of_time_steps,
                number_of_noise_realisations,
            ),
        )

        self.assertEqual(noise.device.type, DEVICE.type)

        fft_of_generated_noise = torch.fft.fft(noise, dim=1, norm="forward")
        fft_of_gt_noise = torch.fft.fft(n1_noise, dim=1, norm="forward")

        psd_of_generated_noise = torch.abs(fft_of_generated_noise) ** 2
        psd_of_gt_noise = torch.abs(fft_of_gt_noise) ** 2

        averaged_psd = torch.mean(psd_of_generated_noise, dim=-1)
        averaged_gt_psd = torch.mean(psd_of_gt_noise, dim=(0, -1))

        abs_diff = torch.abs(averaged_psd - averaged_gt_psd)

        self.assertTrue((abs_diff.max() < 1e-1).item())
        self.assertTrue((abs_diff.mean() < 1e-3).item())

        if plot_psd:
            plt.title("PSD of noise profile 1 GT and generated")
            plt.plot(
                frequencies[frequencies > 0].cpu(),
                averaged_psd[frequencies > 0].cpu(),
                label="generated",
            )
            plt.plot(
                frequencies[frequencies > 0].cpu(),
                averaged_gt_psd[frequencies > 0].cpu(),
                label="gt",
            )
            plt.legend()
            plt.savefig(f"{figure_path}/noise_profile_1_psd")

    def test_noise_profile_5(self):
        plot_psd = False
        total_time = 1
        number_of_time_steps = 1024
        number_of_noise_realisations = 2000
        batch_size = NUM_EXAMPLES_FOR_TEST

        data = load_Vo_dataset(
            data_path,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["noise"],
        )

        frequencies = torch.fft.fftfreq(
            n=number_of_time_steps, d=total_time / number_of_time_steps
        ).to(DEVICE)

        gt_noise = data[0].squeeze().to(DEVICE)
        n5_noise = gt_noise[:, :, :, 1]

        n5_spectral_density = generate_spectal_density_noise_profile_five(
            frequencies=frequencies,
            alpha=1,
        )

        self.assertEqual(
            n5_spectral_density.shape,
            (number_of_time_steps // 2,),
        )

        self.assertEqual(n5_spectral_density.device.type, DEVICE.type)

        n5_sqrt_scaled_spectral_density = (
            generate_sqrt_scaled_spectral_density(
                total_time=total_time,
                spectral_density=n5_spectral_density,
                number_of_time_steps=number_of_time_steps,
            )
        )

        self.assertEqual(
            n5_sqrt_scaled_spectral_density.shape,
            (number_of_time_steps // 2,),
        )

        self.assertEqual(
            n5_sqrt_scaled_spectral_density.device.type, DEVICE.type
        )

        noise = gerenate_noise_time_series_with_one_on_f_noise(
            sqrt_scaled_spectral_density=n5_sqrt_scaled_spectral_density,
            number_of_noise_realisations=number_of_noise_realisations,
            number_of_time_steps=number_of_time_steps,
            batch_size=batch_size,
        )

        self.assertEqual(
            noise.shape,
            (
                batch_size,
                number_of_time_steps,
                number_of_noise_realisations,
            ),
        )

        self.assertEqual(noise.device.type, DEVICE.type)

        fft_of_generated_noise = torch.fft.fft(noise, dim=1, norm="forward")
        fft_of_gt_noise = torch.fft.fft(n5_noise, dim=1, norm="forward")

        psd_of_generated_noise = torch.abs(fft_of_generated_noise) ** 2
        psd_of_gt_noise = torch.abs(fft_of_gt_noise) ** 2

        averaged_psd = torch.mean(psd_of_generated_noise, dim=-1)
        averaged_gt_psd = torch.mean(psd_of_gt_noise, dim=(0, -1))

        abs_diff = torch.abs(averaged_psd - averaged_gt_psd)
        self.assertTrue((abs_diff.max() < 1e-1).item())
        self.assertTrue((abs_diff.mean() < 1e-3).item())

        if plot_psd:
            plt.title("PSD of noise profile 5 GT and generated")
            plt.plot(
                frequencies[frequencies > 0].cpu(),
                averaged_psd[frequencies > 0].cpu(),
                label="generated",
            )
            plt.plot(
                frequencies[frequencies > 0].cpu(),
                averaged_gt_psd[frequencies > 0].cpu(),
                label="gt",
            )
            plt.legend()
            plt.savefig(f"{figure_path}/noise_profile_5_psd")

    def test_noise_profile_3(self):
        plot_noise_correlation = True
        total_time = 1
        number_of_time_steps = 256
        number_of_noise_realisations = 2000
        batch_size = NUM_EXAMPLES_FOR_TEST
        time_step = 0.5 * total_time / number_of_time_steps
        division_factor = 4

        data = load_Vo_dataset(
            data_path2,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["noise"],
        )

        gt_noise = data[0].squeeze().to(DEVICE)
        n3_noise = gt_noise[:, :, :, 0]

        time_range = torch.linspace(
            time_step, total_time - time_step, number_of_time_steps
        ).to(DEVICE)

        time_domain_signal = generate_time_domain_signal_for_noise(
            time_range=time_range,
            total_time=total_time,
        )

        self.assertEqual(
            time_domain_signal.shape,
            (number_of_time_steps,),
        )

        self.assertEqual(time_domain_signal.device.type, DEVICE.type)

        colour_filter = generate_colour_filter_for_noise(
            number_of_time_steps=number_of_time_steps,
            division_factor=division_factor,
        )

        self.assertEqual(
            colour_filter.shape,
            (number_of_time_steps // division_factor,),
        )

        self.assertEqual(colour_filter.device.type, DEVICE.type)

        noise = generate_non_stationary_colour_gaussian_noise_time_series(
            time_domain_signal=time_domain_signal,
            colour_filter=colour_filter,
            number_of_time_steps=number_of_time_steps,
            number_of_noise_realisations=number_of_noise_realisations,
            batch_size=batch_size,
            division_factor=division_factor,
            g=0.2,
        )

        self.assertEqual(
            noise.shape,
            (
                batch_size,
                number_of_time_steps,
                number_of_noise_realisations,
            ),
        )

        self.assertEqual(noise.device.type, DEVICE.type)

        noise_correlation = (torch.bmm(noise, noise.transpose(1, 2))).mean(
            dim=0
        ) / number_of_noise_realisations

        print(n3_noise.shape)

        gt_noise_correlation = (
            torch.bmm(n3_noise, n3_noise.transpose(1, 2))
        ).mean(dim=0) / number_of_noise_realisations

        print(gt_noise_correlation.shape)

        # abs_diff = torch.abs(noise_correlation - gt_noise_correlation)

        # self.assertTrue(
        #     (abs_diff.max() < 1.0),
        # )
        # self.assertTrue(
        #     (abs_diff.mean() < 0.1),
        # )

        if plot_noise_correlation:
            plt.figure(figsize=(10, 10))
            plt.imshow(gt_noise_correlation.cpu())
            plt.title("GT N3 Correlation Matrix")
            plt.xlabel("Time Steps")
            plt.ylabel("Time Steps")
            plt.savefig(f"{figure_path}/gt_noise_profile_3_correlation_{number_of_time_steps}.png")

            plt.figure(figsize=(10, 10))
            plt.imshow(noise_correlation.cpu())
            plt.title("N3 Correlation Matrix")
            plt.xlabel("Time Steps")
            plt.ylabel("Time Steps")
            plt.savefig(f"{figure_path}/noise_profile_3_correlation_{number_of_time_steps}.png")

    def test_noise_profile_6(self):
        plot_noise_correlation = False
        data = load_Vo_dataset(
            data_path2,
            num_examples=NUM_EXAMPLES_FOR_TEST,
            data_of_interest=["noise"],
        )

        gt_noise = data[0].squeeze().to(DEVICE)
        n3_noise = gt_noise[:, :, :, 0].to(DEVICE)
        n6_noise = gt_noise[:, :, :, 1]

        noise = square_and_scale_noise_time_series(
            noise_time_series=n3_noise,
            g=0.3,
        )

        self.assertEqual(noise.shape, n6_noise.shape)

        self.assertEqual(noise.device.type, DEVICE.type)

        self.assertTrue(torch.allclose(noise, n6_noise))

        noise_correlation = (torch.bmm(noise, noise.transpose(1, 2))).mean(
            dim=0
        ) / 2000

        gt_noise_correlation = (
            torch.bmm(n6_noise, n6_noise.transpose(1, 2))
        ).mean(dim=0) / 2000

        if plot_noise_correlation:
            plt.figure(figsize=(10, 10))
            plt.imshow(gt_noise_correlation.cpu())
            plt.title("GT N6 Correlation Matrix")
            plt.xlabel("Time Steps")
            plt.ylabel("Time Steps")
            plt.savefig(f"{figure_path}/gt_noise_profile_6_correlation.png")

            plt.figure(figsize=(10, 10))
            plt.imshow(noise_correlation.cpu())
            plt.title("N6 Correlation Matrix")
            plt.xlabel("Time Steps")
            plt.ylabel("Time Steps")
            plt.savefig(f"{figure_path}/noise_profile_6_correlation.png")
