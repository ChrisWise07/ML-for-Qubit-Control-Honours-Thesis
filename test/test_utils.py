import unittest
import torch
import math
import itertools
from time_series_to_noise.utils import (
    compute_nuc_norm_of_diff_between_batch_of_matrices,
    fidelity_batch_of_matrices,
    compute_mean_distance_matrix_for_batch_of_VO_matrices,
    calculate_trig_expo_funcs_for_batch,
    return_estimated_VO_unital_for_batch,
    calculate_ground_turth_parameters,
    calculate_expectation_values,
    calculate_xyz_vo_from_expectation_values_wrapper,
    load_qdataset,
    load_Vo_dataset_zip,
    calculate_state_from_observable_expectations,
    compute_process_matrix_for_single_qubit,
    compute_process_fidelity,
)




SIGMA_X = [[0, 1], [1, 0]]
SIGMA_Y = [[0, -1j], [1j, 0]]
SIGMA_Z = [[1, 0], [0, -1]]


RHO = torch.tensor(
    [
        [[0.5, 0], [0, 0.5]],
        [[1, 0], [0, 0]],
        [[0.4j, 0.0], [1.3j, 2.0]],
        [[0.4, 0.2], [1.3, 2.0]],
    ]
)

SIGMA = torch.tensor(
    [
        [[0.5, 0], [0, 0.5]],
        [[0, 0], [0, 1]],
        [[0.4, 0.0], [1.3, 2.0j]],
        [[0.6j, 0.3j], [1.2j, 1.75j]],
    ]
)

PATH_TO_DATA_FOLDER = "/home/chriswise/github/Honours-Research-ML-for-QC/QData_pickle/G_1q_XY_XZ_N3N6_comb"

NUM_EXAMPLES_TO_TEST = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_ANGLE_THRESHOLDS = torch.tensor(
    [
        torch.pi / 2,
        torch.pi / 2,
        1.0,
        torch.pi / 2,
        torch.pi / 2,
        1.0,
        torch.pi / 2,
        torch.pi / 2,
        1.0,
    ]
)

MIN_ANGLE_THRESHOLDS = torch.tensor(
    [
        -torch.pi / 2,
        0.0,
        0.0,
        -torch.pi / 2,
        0.0,
        0.0,
        -torch.pi / 2,
        0.0,
        0.0,
    ]
)

SIGMA_I_GATE = torch.tensor(
    [[1, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE
)

SIGMA_X_GATE = torch.tensor(
    [[0, 1], [1, 0]], dtype=torch.cfloat, device=DEVICE
)

SIGMA_Y_GATE = torch.tensor(
    [[0, -1j], [1j, 0]], dtype=torch.cfloat, device=DEVICE
)

SIGMA_Z_GATE = torch.tensor(
    [[1, 0], [0, -1]], dtype=torch.cfloat, device=DEVICE
)

H_GATE = torch.tensor(
    [
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), -1 / math.sqrt(2)],
    ],
    dtype=torch.cfloat,
    device=DEVICE,
)

R_X_PI_ON_FOUR_GATE = torch.tensor(
    [
        [math.cos(math.pi / 8), -1j * math.sin(math.pi / 8)],
        [-1j * math.sin(math.pi / 8), math.cos(math.pi / 8)],
    ],
    dtype=torch.cfloat,
    device=DEVICE,
)

RHO_ZERO = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat, device=DEVICE)

RHO_ONE = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE)

RHO_PLUS = torch.tensor(
    [[0.5, 0.5], [0.5, 0.5]], dtype=torch.cfloat, device=DEVICE
)

RHO_MINUS = torch.tensor(
    [[0.5, -0.5], [-0.5, 0.5]], dtype=torch.cfloat, device=DEVICE
)


class TestTraceDistance(unittest.TestCase):
    def test_trace_distance_batch_of_matrices(self):
        expected = torch.tensor([0.0, 2.0, 3.86005, 3.66062])

        actual = compute_nuc_norm_of_diff_between_batch_of_matrices(RHO, SIGMA)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestFidelity(unittest.TestCase):
    def test_fidelity_batch_of_matrices(self):
        expected = torch.tensor([1.0, 0.0, 0.135072, 0.984895])

        actual = fidelity_batch_of_matrices(RHO, SIGMA)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestTraceDistanceBasedLoss(unittest.TestCase):
    def test_trace_distance_based_loss_equal_matrcies(self):
        estimated_VX = torch.tensor([SIGMA_X] * 3, dtype=torch.complex64)
        estimated_VY = torch.tensor([SIGMA_Y] * 3, dtype=torch.complex64)
        estimated_VZ = torch.tensor([SIGMA_Z] * 3, dtype=torch.complex64)
        VX_true = estimated_VX.clone()
        VY_true = estimated_VY.clone()
        VZ_true = estimated_VZ.clone()
        self.assertEqual(
            compute_mean_distance_matrix_for_batch_of_VO_matrices(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            0.0,
        )

    def test_trace_distance_based_loss_max(self):
        estimated_VX = torch.tensor([SIGMA_Y] * 3, dtype=torch.cfloat)
        estimated_VY = torch.tensor([SIGMA_Z] * 3, dtype=torch.cfloat)
        estimated_VZ = torch.tensor([SIGMA_X] * 3, dtype=torch.cfloat)
        VX_true = torch.tensor([SIGMA_X] * 3, dtype=torch.cfloat)
        VY_true = torch.tensor([SIGMA_Y] * 3, dtype=torch.cfloat)
        VZ_true = torch.tensor([SIGMA_Z] * 3, dtype=torch.cfloat)
        self.assertAlmostEqual(
            compute_mean_distance_matrix_for_batch_of_VO_matrices(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            2 * torch.sqrt(torch.tensor(2.0)).item(),
            places=5,
        )

    def test_trace_distance_based_loss_mixture(self):
        estimated_VX = RHO
        estimated_VY = RHO.roll(1, 0)
        estimated_VZ = RHO.roll(2, 0)
        VX_true = SIGMA
        VY_true = SIGMA.roll(1, 0)
        VZ_true = SIGMA.roll(2, 0)
        self.assertAlmostEqual(
            compute_mean_distance_matrix_for_batch_of_VO_matrices(
                estimated_VX,
                estimated_VY,
                estimated_VZ,
                VX_true,
                VY_true,
                VZ_true,
            ).item(),
            2.38017,
            places=5,
        )


class TestCalculateTrigExpoFuncsForBatch(unittest.TestCase):
    def test_calculate_trig_expo_funcs_for_batch_size_1(self):
        theta = torch.tensor([2 * torch.pi])
        psi = torch.tensor([0.0])

        expected = torch.tensor(
            [
                [
                    [1, 0 + 0j],
                    [0 + 0j, -1],
                ]
            ]
        )

        actual = calculate_trig_expo_funcs_for_batch(psi, theta)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_calculate_trig_expo_funcs_for_batch_size_2(self):
        theta = torch.tensor([2 * torch.pi, 0.0])
        psi = torch.tensor([0.0, 2 * torch.pi])

        expected = torch.tensor(
            [
                [
                    [1, 0 + 0j],
                    [0 + 0j, -1],
                ],
                [
                    [1, 0],
                    [0, -1],
                ],
            ]
        )

        actual = calculate_trig_expo_funcs_for_batch(psi, theta)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_calculate_trig_expo_funcs_for_batch_size_3(self):
        theta = torch.tensor([2 * torch.pi, 0.0, 1.34])
        psi = torch.tensor([0.0, 2 * torch.pi, 5.4])

        expected = torch.tensor(
            [
                [
                    [1, 0 + 0j],
                    [0 + 0j, -1],
                ],
                [
                    [1, 0],
                    [0, -1],
                ],
                [
                    [-0.895344, 0.0865496 + 0.436884j],
                    [0.0865496 - 0.436884j, 0.895344],
                ],
            ]
        )

        actual = calculate_trig_expo_funcs_for_batch(psi, theta)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class TestReturnEstimatedVOUnitalForBatch9Params(unittest.TestCase):
    def test_return_estimated_VO_unital_for_batch(self):
        batch_parameters = torch.tensor(
            [
                [
                    torch.pi / 2,
                    0.0,
                    1.0,
                    2.7,
                    5.4,
                    0.7,
                    1 / 2,
                    torch.pi / 2,
                    0.0,
                ],
                [
                    1 / 2,
                    torch.pi / 2,
                    0.0,
                    torch.pi / 2,
                    0.0,
                    1.0,
                    2.7,
                    5.4,
                    0.7,
                ],
                [
                    2.7,
                    5.4,
                    0.7,
                    1 / 2,
                    torch.pi / 2,
                    0.0,
                    torch.pi / 2,
                    0.0,
                    1.0,
                ],
            ],
            dtype=torch.float32,
            device=DEVICE,
        )
        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(batch_parameters)

        expected_VX = torch.tensor(
            [
                [
                    [0.0 + 0.0j, -1.0 + 0.0j],
                    [1.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 - 0.0j],
                ],
                [
                    [0.4358 + 0.5306j, 0.136 + 0.0j],
                    [-0.136 + 0.0j, 0.4358 - 0.5306j],
                ],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        expected_VY = torch.tensor(
            [
                [
                    [0.5306 - 0.4358j, 0.0 - 0.136j],
                    [0.0 - 0.136j, 0.5306 + 0.4358j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 1.0j],
                    [0.0 + 1.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 - 0.0j],
                ],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        expected_VZ = torch.tensor(
            [
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                [
                    [-0.136 + 0.0j, 0.4358 - 0.5306j],
                    [-0.4358 - 0.5306j, -0.136 + 0.0j],
                ],
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        self.assertEqual(estimated_VX.shape, (3, 2, 2))
        self.assertEqual(estimated_VY.shape, (3, 2, 2))
        self.assertEqual(estimated_VZ.shape, (3, 2, 2))

        self.assertTrue(
            torch.allclose(
                estimated_VX,
                expected_VX,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VY,
                expected_VY,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VZ,
                expected_VZ,
                atol=1e-04,
            )
        )


class TestReturnEstimatedVOUnitalForBatch12Params(unittest.TestCase):
    def test_return_estimated_VO_unital_for_batch_12(self):
        batch_parameters = torch.tensor(
            [
                [
                    torch.pi / 3,
                    torch.pi / 3,
                    torch.pi / 3,
                    1.0,
                    2.7,
                    5.4,
                    1.5,
                    0.7,
                    torch.pi / 2,
                    torch.pi / 2,
                    torch.pi / 2,
                    0.0,
                ],
                [
                    torch.pi / 2,
                    torch.pi / 2,
                    torch.pi / 2,
                    0.0,
                    torch.pi / 3,
                    torch.pi / 3,
                    torch.pi / 3,
                    1.0,
                    2.7,
                    5.4,
                    1.5,
                    0.7,
                ],
                [
                    2.7,
                    5.4,
                    1.5,
                    0.7,
                    torch.pi / 2,
                    torch.pi / 2,
                    torch.pi / 2,
                    0.0,
                    torch.pi / 3,
                    torch.pi / 3,
                    torch.pi / 3,
                    1.0,
                ],
            ],
            dtype=torch.float32,
        ).to(DEVICE)

        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(batch_parameters)

        expected_VX = torch.tensor(
            [
                [
                    [0.43301 + 0.75j, 0.5 + 0j],
                    [-0.5 + 0j, 0.43301 - 0.75j],
                ],
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
                [
                    [0.43582 + 0.53062j, 0.13603 + 0j],
                    [-0.13603 + 0j, 0.43582 - 0.53062j],
                ],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        expected_VY = torch.tensor(
            [
                [
                    [0.53062 - 0.43582j, 0 - 0.13603j],
                    [0 - 0.13603j, 0.53062 + 0.43582j],
                ],
                [
                    [0.75 - 0.43301j, 0 - 0.5j],
                    [0 - 0.5j, 0.75 + 0.43301j],
                ],
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        expected_VZ = torch.tensor(
            [
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
                [
                    [-0.13603 + 0j, 0.43582 - 0.53062j],
                    [-0.43582 - 0.53062j, -0.13603 + 0j],
                ],
                [
                    [-0.5 + 0j, 0.43301 - 0.75j],
                    [-0.43301 - 0.75j, -0.5 + 0j],
                ],
            ],
            dtype=torch.cfloat,
            device=DEVICE,
        )

        self.assertEqual(estimated_VX.shape, (3, 2, 2))
        self.assertEqual(estimated_VY.shape, (3, 2, 2))
        self.assertEqual(estimated_VZ.shape, (3, 2, 2))

        self.assertTrue(
            torch.allclose(
                estimated_VX,
                expected_VX,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VY,
                expected_VY,
                atol=1e-04,
            )
        )
        self.assertTrue(
            torch.allclose(
                estimated_VZ,
                expected_VZ,
                atol=1e-04,
            )
        )


class TestReturnEstimatedParametersForBatch(unittest.TestCase):
    def test_return_parameters_for_batch(self):
        atol = 1e-04
        strict_atol = 1e-06

        ground_truth_VX = torch.tensor(
            [
                [[0.433013 + 0.75j, 0.5 + 0j], [-0.5 + 0j, 0.433013 - 0.75j]],
                [[-0.00841471, -0.00540302], [0.00540302, -0.00841471]],
                [
                    [0.0369154 + 0.044946j, -0.697579 + 0j],
                    [0.697579 + 0j, 0.0369154 - 0.044946j],
                ],
                [
                    [0.43301 + 0.75j, 0.5 + 0j],
                    [-0.5 + 0j, 0.43301 - 0.75j],
                ],
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
                [
                    [0.43582 + 0.53062j, 0.13603 + 0j],
                    [-0.13603 + 0j, 0.43582 - 0.53062j],
                ],
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        ground_truth_VY = torch.tensor(
            [
                [
                    [0.04495 - 0.03692j, 0 + 0.6976j],
                    [0 + 0.6976j, 0.04495 + 0.03692j],
                ],
                [[0.75 - 0.433j, 0 - 0.5j], [0 - 0.5j, 0.75 + 0.433j]],
                [
                    [0 + 0.008415j, 0 + 0.005403j],
                    [0 + 0.005403j, 0 - 0.008415j],
                ],
                [
                    [0.53062 - 0.43582j, 0 - 0.13603j],
                    [0 - 0.13603j, 0.53062 + 0.43582j],
                ],
                [
                    [0.75 - 0.43301j, 0 - 0.5j],
                    [0 - 0.5j, 0.75 + 0.43301j],
                ],
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        ground_truth_VZ = torch.tensor(
            [
                [[0.005403, -0.008415], [0.008415, 0.005403]],
                [[0.6976, 0.03692 - 0.04495j], [-0.03692 - 0.04495j, 0.6976]],
                [[-0.5, 0.433 - 0.75j], [-0.433 - 0.75j, -0.5]],
                [
                    [0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j],
                ],
                [
                    [-0.13603 + 0j, 0.43582 - 0.53062j],
                    [-0.43582 - 0.53062j, -0.13603 + 0j],
                ],
                [
                    [-0.5 + 0j, 0.43301 - 0.75j],
                    [-0.43301 - 0.75j, -0.5 + 0j],
                ],
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        calculated_parameters = calculate_ground_turth_parameters(
            ground_truth_VX,
            ground_truth_VY,
            ground_truth_VZ,
        )

        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(calculated_parameters)

        self.assertTrue(
            torch.all(
                torch.le(
                    torch.round(
                        torch.max(calculated_parameters, dim=0).values,
                        decimals=6,
                    ),
                    MAX_ANGLE_THRESHOLDS.to(DEVICE),
                )
            )
        )

        self.assertTrue(
            torch.all(
                torch.ge(
                    torch.round(
                        torch.min(calculated_parameters, dim=0).values,
                        decimals=6,
                    ),
                    MIN_ANGLE_THRESHOLDS.to(DEVICE),
                )
            )
        )

        self.assertTrue(
            torch.allclose(
                estimated_VX,
                ground_truth_VX,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(torch.abs(estimated_VX - ground_truth_VX)).item()
            < strict_atol
        )

        self.assertTrue(
            torch.allclose(
                estimated_VY,
                ground_truth_VY,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(torch.abs(estimated_VY - ground_truth_VY)).item()
            < strict_atol
        )

        self.assertTrue(
            torch.allclose(
                estimated_VZ,
                ground_truth_VZ,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(torch.abs(estimated_VZ - ground_truth_VZ)).item()
            < strict_atol
        )

    def test_parameter_construction_for_dataset(self):
        atol = 1e-04
        strict_atol = 1e-05

        data = load_qdataset(
            path_to_dataset=PATH_TO_DATA_FOLDER,
            num_examples=NUM_EXAMPLES_TO_TEST,
        )

        ground_truth_VX_for_test = data["Vx"].to(DEVICE)
        ground_truth_VY_for_test = data["Vy"].to(DEVICE)
        ground_truth_VZ_for_test = data["Vz"].to(DEVICE)

        calculated_parameters = calculate_ground_turth_parameters(
            ground_truth_VX_for_test,
            ground_truth_VY_for_test,
            ground_truth_VZ_for_test,
        )

        (
            estimated_VX_for_test,
            estimated_VY_for_test,
            estimated_VZ_for_test,
        ) = return_estimated_VO_unital_for_batch(calculated_parameters)

        self.assertTrue(
            torch.all(
                torch.max(calculated_parameters, dim=0).values
                <= MAX_ANGLE_THRESHOLDS.to(DEVICE)
            )
        )

        self.assertTrue(
            torch.all(
                torch.min(calculated_parameters, dim=0).values
                >= MIN_ANGLE_THRESHOLDS.to(DEVICE)
            )
        )

        self.assertTrue(
            torch.allclose(
                estimated_VX_for_test,
                ground_truth_VX_for_test,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(
                torch.abs(estimated_VX_for_test - ground_truth_VX_for_test)
            ).item()
            < strict_atol
        )

        self.assertTrue(
            torch.allclose(
                estimated_VY_for_test,
                ground_truth_VY_for_test,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(
                torch.abs(estimated_VY_for_test - ground_truth_VY_for_test)
            ).item()
            < strict_atol
        )

        self.assertTrue(
            torch.allclose(
                estimated_VZ_for_test,
                ground_truth_VZ_for_test,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.mean(
                torch.abs(estimated_VZ_for_test - ground_truth_VZ_for_test)
            ).item()
            < strict_atol
        )


class TestExpectationConstruction(unittest.TestCase):
    def test_expectation_construction(self):
        data = load_qdataset(
            path_to_dataset=PATH_TO_DATA_FOLDER,
            num_examples=NUM_EXAMPLES_TO_TEST,
        )

        Vo_operators = data["Vo"].transpose(0, 1).to(DEVICE)

        calculated_expectation_values = calculate_expectation_values(
            Vo_operators=Vo_operators,
            control_unitaries=data["control_unitaries"].to(DEVICE),
        )

        self.assertTrue(
            torch.allclose(
                calculated_expectation_values,
                data["expectations"].to(DEVICE),
                atol=1e-04,
            )
        )

    def test_expectation_construction_two_qubits(self):
        two_qubit_data_path = "/home/chriswise/github/Honours-Research-ML-for-QC/juypter_notebooks/G_2q_IX-XI-XX"

        data = load_Vo_dataset_zip(
            path_to_dataset=two_qubit_data_path,
            num_examples=30,
            data_of_interest=["expectations", "Vo_operator", "U0"],
        )

        ground_turth_expectations = data[0].squeeze().to(DEVICE)
        Vo_operators = data[1].transpose(0, 1).squeeze().to(DEVICE)
        control_unitaries = data[2].squeeze()[:, -1].to(DEVICE)

        calculated_expectation_values = calculate_expectation_values(
            Vo_operators=Vo_operators,
            control_unitaries=control_unitaries,
        )

        self.assertTrue(
            torch.allclose(
                calculated_expectation_values,
                ground_turth_expectations,
                atol=1e-04,
            )
        )


class TestVoConstructionFromExpectationValuesForBatch(unittest.TestCase):
    def test_Vo_construction_from_expectation_values_for_batch(self):
        atol = 1e-04

        data = load_qdataset(
            path_to_dataset=PATH_TO_DATA_FOLDER,
            num_examples=1000,
        )

        Vx = data["Vo"][:, 0].to(DEVICE)
        Vy = data["Vo"][:, 1].to(DEVICE)
        Vz = data["Vo"][:, 2].to(DEVICE)

        expectations = data["expectations"].to(DEVICE)
        control_unitaries = data["control_unitaries"].to(DEVICE)
        import time
        import numpy as np
        timings = []
        print("Timing Vo construction")
        for i in range(6):
            start = time.time()
            (
                new_Vx,
                new_Vy,
                new_Vz,
            ) = calculate_xyz_vo_from_expectation_values_wrapper(
                expectation_values=expectations,
                control_unitaries=control_unitaries,
            )
            end = time.time()
            timings.append(end - start)
        print(timings)
        print("Mean time: ", np.mean(timings[1:]))
        print("Std time: ", np.std(timings[1:]))


        torch.set_printoptions(precision=6, sci_mode=False)

        print(new_Vx.shape)
        print(Vx.shape)

        vx_diff = torch.abs(new_Vx - Vx)
        vy_diff = torch.abs(new_Vy - Vy)
        vz_diff = torch.abs(new_Vz - Vz)

        print("Vx diff max: ", torch.max(vx_diff))
        print("Vy diff max: ", torch.max(vy_diff))
        print("Vz diff max: ", torch.max(vz_diff))

        print("Vx diff mean: ", torch.mean(vx_diff))
        print("Vy diff mean: ", torch.mean(vy_diff))
        print("Vz diff mean: ", torch.mean(vz_diff))

        self.assertTrue(
            torch.allclose(
                new_Vx,
                Vx,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.allclose(
                new_Vy,
                Vy,
                atol=atol,
            )
        )

        self.assertTrue(
            torch.allclose(
                new_Vz,
                Vz,
                atol=atol,
            )
        )


class TestStateTomographyConstruction(unittest.TestCase):
    def test_state_tomography_all(self):
        density_matrices_list = [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            0.5 * torch.tensor([[1, 1], [1, 1]]),
            0.5 * torch.tensor([[1, -1], [-1, 1]]),
            0.5 * torch.tensor([[1, -1j], [1j, 1]]),
            0.5 * torch.tensor([[1, 1j], [-1j, 1]]),
            0.5 * torch.tensor([[1, 0], [0, 1]]),
        ]

        # Convert the list to a PyTorch tensor
        expected_density_matrices = torch.tensor(
            density_matrices_list, dtype=torch.cfloat, device=DEVICE
        )

        expectation_values = [
            [0, 0, 1],
            [0, 0, -1],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
        ]

        # Convert the list to a PyTorch tensor
        expectation_values_tensor = torch.tensor(
            expectation_values, dtype=torch.float, device=DEVICE
        )

        observables_repeated = (
            torch.tensor([SIGMA_X, SIGMA_Y, SIGMA_Z], dtype=torch.cfloat)
            .repeat(7, 1, 1, 1)
            .to(DEVICE)
        )

        identity = SIGMA_I.unsqueeze(0).repeat(7, 1, 1).to(DEVICE)

        rho = calculate_state_from_observable_expectations(
            expectation_values=expectation_values_tensor,
            observables=observables_repeated,
            identity=identity,
        )

        self.assertEqual(rho.shape, (7, 2, 2))

        self.assertTrue(torch.allclose(rho, expected_density_matrices))


def apply_gate(rho: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (gate @ rho @ gate.conj().transpose(-2, -1)).unsqueeze(0)


class TestProcessTomography(unittest.TestCase):
    def test_process_matrix(self):
        expected_chi_matrices = torch.tensor(
            [
                [
                    [
                        0.7500 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 - 0.2500j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.2500 + 0.0000j,
                        0.0000 - 0.2500j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.0000 + 0.2500j,
                        -0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.2500j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.2500 + 0.0000j,
                    ],
                ],
                [
                    [
                        0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 - 0.2500j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.7500 + 0.0000j,
                        0.0000 - 0.2500j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.0000 + 0.2500j,
                        0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.2500j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        -0.2500 + 0.0000j,
                    ],
                ],
                [
                    [
                        -0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.2500j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.2500 + 0.0000j,
                        0.0000 + 0.2500j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.0000 - 0.2500j,
                        0.7500 + 0.0000j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 - 0.2500j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.2500 + 0.0000j,
                    ],
                ],
                [
                    [
                        0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.2500j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        -0.2500 + 0.0000j,
                        0.0000 + 0.2500j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.0000 - 0.2500j,
                        0.2500 + 0.0000j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 - 0.2500j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.0000j,
                        0.7500 + 0.0000j,
                    ],
                ],
                [
                    [
                        0.2500 + 0.0000j,
                        0.0000 + 0.2500j,
                        -0.0000 + 0.0000j,
                        0.0000 - 0.0000j,
                    ],
                    [
                        0.0000 - 0.2500j,
                        0.2500 + 0.0000j,
                        0.0000 - 0.0000j,
                        0.5000 + 0.0000j,
                    ],
                    [
                        -0.0000 - 0.0000j,
                        0.0000 + 0.0000j,
                        0.2500 + 0.0000j,
                        -0.0000 + 0.2500j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.5000 - 0.0000j,
                        -0.0000 - 0.2500j,
                        0.2500 + 0.0000j,
                    ],
                ],
                [
                    [
                        0.6768 + 0.0000j,
                        0.0000 + 0.1768j,
                        0.0000 + 0.0000j,
                        0.0000 - 0.2500j,
                    ],
                    [
                        0.0000 - 0.1768j,
                        0.3232 + 0.0000j,
                        0.0000 - 0.2500j,
                        0.0000 + 0.0000j,
                    ],
                    [
                        0.0000 + 0.0000j,
                        0.0000 + 0.2500j,
                        -0.1768 + 0.0000j,
                        0.0000 - 0.1768j,
                    ],
                    [
                        0.0000 + 0.2500j,
                        0.0000 + 0.0000j,
                        0.0000 + 0.1768j,
                        0.1768 + 0.0000j,
                    ],
                ],
            ],
            device=DEVICE,
            dtype=torch.cfloat,
        )

        rho_zeros = torch.cat(
            [
                apply_gate(RHO_ZERO, SIGMA_I_GATE),
                apply_gate(RHO_ZERO, SIGMA_X_GATE),
                apply_gate(RHO_ZERO, SIGMA_Y_GATE),
                apply_gate(RHO_ZERO, SIGMA_Z_GATE),
                apply_gate(RHO_ZERO, H_GATE),
                apply_gate(RHO_ZERO, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_ones = torch.cat(
            [
                apply_gate(RHO_ONE, SIGMA_I_GATE),
                apply_gate(RHO_ONE, SIGMA_X_GATE),
                apply_gate(RHO_ONE, SIGMA_Y_GATE),
                apply_gate(RHO_ONE, SIGMA_Z_GATE),
                apply_gate(RHO_ONE, H_GATE),
                apply_gate(RHO_ONE, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_pulses = torch.cat(
            [
                apply_gate(RHO_PLUS, SIGMA_I_GATE),
                apply_gate(RHO_PLUS, SIGMA_X_GATE),
                apply_gate(RHO_PLUS, SIGMA_Y_GATE),
                apply_gate(RHO_PLUS, SIGMA_Z_GATE),
                apply_gate(RHO_PLUS, H_GATE),
                apply_gate(RHO_PLUS, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_minuses = torch.cat(
            [
                apply_gate(RHO_MINUS, SIGMA_I_GATE),
                apply_gate(RHO_MINUS, SIGMA_X_GATE),
                apply_gate(RHO_MINUS, SIGMA_Y_GATE),
                apply_gate(RHO_MINUS, SIGMA_Z_GATE),
                apply_gate(RHO_MINUS, H_GATE),
                apply_gate(RHO_MINUS, R_X_PI_ON_FOUR_GATE),
            ],
        )

        chi_matrices = compute_process_matrix_for_single_qubit(
            rho_zeros, rho_ones, rho_pulses, rho_minuses
        )

        self.assertTrue(
            torch.allclose(
                chi_matrices,
                expected_chi_matrices,
                atol=1e-04,
            )
        )

    def test_process_fidelity(self):
        rho_zeros = torch.cat(
            [
                apply_gate(RHO_ZERO, SIGMA_I_GATE),
                apply_gate(RHO_ZERO, SIGMA_X_GATE),
                apply_gate(RHO_ZERO, SIGMA_Y_GATE),
                apply_gate(RHO_ZERO, SIGMA_Z_GATE),
                apply_gate(RHO_ZERO, H_GATE),
                apply_gate(RHO_ZERO, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_ones = torch.cat(
            [
                apply_gate(RHO_ONE, SIGMA_I_GATE),
                apply_gate(RHO_ONE, SIGMA_X_GATE),
                apply_gate(RHO_ONE, SIGMA_Y_GATE),
                apply_gate(RHO_ONE, SIGMA_Z_GATE),
                apply_gate(RHO_ONE, H_GATE),
                apply_gate(RHO_ONE, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_pulses = torch.cat(
            [
                apply_gate(RHO_PLUS, SIGMA_I_GATE),
                apply_gate(RHO_PLUS, SIGMA_X_GATE),
                apply_gate(RHO_PLUS, SIGMA_Y_GATE),
                apply_gate(RHO_PLUS, SIGMA_Z_GATE),
                apply_gate(RHO_PLUS, H_GATE),
                apply_gate(RHO_PLUS, R_X_PI_ON_FOUR_GATE),
            ],
        )

        rho_minuses = torch.cat(
            [
                apply_gate(RHO_MINUS, SIGMA_I_GATE),
                apply_gate(RHO_MINUS, SIGMA_X_GATE),
                apply_gate(RHO_MINUS, SIGMA_Y_GATE),
                apply_gate(RHO_MINUS, SIGMA_Z_GATE),
                apply_gate(RHO_MINUS, H_GATE),
                apply_gate(RHO_MINUS, R_X_PI_ON_FOUR_GATE),
            ],
        )

        chi_matrices = compute_process_matrix_for_single_qubit(
            rho_zeros, rho_ones, rho_pulses, rho_minuses
        )

        all_pairs = list(
            itertools.combinations_with_replacement(chi_matrices, 2)
        )

        process_matrices_one = torch.stack([pair[0] for pair in all_pairs])
        process_matrices_two = torch.stack([pair[1] for pair in all_pairs])

        expected_fidelity = torch.tensor(
            [
                1.0,
                0.5,
                -0.5,
                0.0,
                0.25,
                0.926777,
                1.0,
                0.0,
                -0.5,
                0.25,
                0.573223,
                1.0,
                0.5,
                0.25,
                -0.426777,
                1.0,
                0.25,
                -0.0732233,
                1.0,
                0.25,
                1.0,
            ],
            dtype=torch.float,
            device=DEVICE,
        )

        fidelity = compute_process_fidelity(
            process_matrix_one=process_matrices_one,
            process_matrix_two=process_matrices_two,
        )

        self.assertTrue(
            torch.allclose(
                fidelity,
                expected_fidelity,
                atol=1e-07,
            )
        )
