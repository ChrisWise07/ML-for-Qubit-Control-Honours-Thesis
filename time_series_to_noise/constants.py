import torch
import math

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HDF5_DIR_PATH = "/mnt/c/Users/ChrisWiseXPSLocal/GitHub/Honours-Research-ML-for-QC/QData_hdf5"

DATA_SET_NAMES = [
    "G_1q_X",
    "G_1q_XY",
    "G_1q_XY_XZ_N1N5",
    "G_1q_XY_XZ_N1N6",
    "G_1q_XY_XZ_N3N6",
    "G_1q_X_Z_N1",
    "G_1q_X_Z_N2",
    "G_1q_X_Z_N3",
    "G_1q_X_Z_N4",
    "G_2q_IX-XI_IZ-ZI_N1-N6",
    "G_2q_IX-XI-XX",
    "G_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "G_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "S_1q_X",
    "S_1q_XY",
    "S_1q_XY_XZ_N1N5",
    "S_1q_XY_XZ_N1N6",
    "S_1q_XY_XZ_N3N6",
    "S_1q_X_Z_N1",
    "S_1q_X_Z_N2",
    "S_1q_X_Z_N3",
    "S_1q_X_Z_N4",
    "S_2q_IX-XI_IZ-ZI_N1-N6",
    "S_2q_IX-XI-XX",
    "S_2q_IX-XI-XX_IZ-ZI_N1-N5",
    "S_2q_IX-XI-XX_IZ-ZI_N1-N6",
]

SIGMA_I = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE)
SIGMA_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=DEVICE)
SIGMA_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=DEVICE)
SIGMA_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=DEVICE)


# Tensor has shape (1, 3, 1, 2, 2) for batched matrix multiplication
COMBINED_SIGMA_TENSOR = (
    torch.stack([SIGMA_X, SIGMA_Y, SIGMA_Z], dim=0).unsqueeze(1).unsqueeze(0)
).to(DEVICE)

# Tensor has shape (1, 3, 1, 4, 4) for batched matrix multiplication
COMBINED_SIGMA_TENSOR_TWO_QUBITS = (
    torch.stack(
        [
            torch.kron(SIGMA_ONE, SIGMA_TWO)
            for SIGMA_ONE in [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]
            for SIGMA_TWO in [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]
        ],
        dim=0,
    )[1:]
    .unsqueeze(1)
    .unsqueeze(0)
).to(DEVICE)

LIST_OF_PAULI_EIGENSTATES = torch.tensor(
    [
        [[1 / 2, 1 / 2], [1 / 2, 1 / 2]],
        [[1 / 2, -1 / 2], [-1 / 2, 1 / 2]],
        [[1 / 2, -1j / 2], [1j / 2, 1 / 2]],
        [[1 / 2, 1j / 2], [-1j / 2, 1 / 2]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
    ],
    dtype=torch.cfloat,
    device=DEVICE,
)

MIXED = (
    1
    / 2
    * (
        SIGMA_I
        + 1 / math.sqrt(3) * SIGMA_X
        + 1 / math.sqrt(3) * SIGMA_Y
        + 1 / math.sqrt(3) * SIGMA_Z
    )
).unsqueeze(0)

LIST_OF_PAULI_EIGENSTATES_TWO_QUBITS = torch.stack(
    [
        torch.kron(STATE_ONE, STATE_TWO)
        for STATE_ONE in LIST_OF_PAULI_EIGENSTATES
        for STATE_TWO in LIST_OF_PAULI_EIGENSTATES
    ],
    dim=0,
).to(DEVICE)

EPSILON = 1e-8

UNIVERSAL_GATE_SET_SINGLE_QUBIT = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat, device=DEVICE),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=DEVICE),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=DEVICE),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=DEVICE),
    "H": torch.tensor(
        [
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ),
    "R_X_PI/4": torch.tensor(
        [
            [math.cos(math.pi / 8), -1j * math.sin(math.pi / 8)],
            [-1j * math.sin(math.pi / 8), math.cos(math.pi / 8)],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ),
}

CNOT = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=torch.cfloat,
    device=DEVICE,
)

LAMBDA_MATRIX_FOR_CHI_MATRIX = (
    1
    / 2
    * torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, -1, 0],
            [1, 0, 0, -1],
        ],
        dtype=torch.cfloat,
        device=DEVICE,
    ).unsqueeze(0)
)


IDEAL_PROCESS_MATRICES = {
    "I": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "X": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "Y": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "Z": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "H": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
    "R_X_PI/4": torch.tensor(
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
        device=DEVICE,
        dtype=torch.cfloat,
    ),
}
