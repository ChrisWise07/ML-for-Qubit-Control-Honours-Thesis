import torch
import pandas as pd
from hc_optimisation_of_pulses import (
    find_optimal_control_pulses,
)

torch.manual_seed(0)

hyperparameters_to_scores = {}
optimal_control_pulse_sequences = None

best_min_fidelity = 0

for sequence_length in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for num_iters in [125, 250, 500, 1000]:
        for std in [0.03125, 0.0625, 0.125, 0.25]:
            for init_sol_type in [
                "noiseless_ideal",
                "uniform_distro",
                "normal_distro",
            ]:
                (
                    fidelity_scores,
                    optimal_pulse_sequences,
                ) = find_optimal_control_pulses(
                    num_iters=num_iters,
                    sequence_length=sequence_length,
                    std=std,
                    init_sol_type=init_sol_type,
                )

                hyperparameters_to_scores[
                    (
                        num_iters,
                        sequence_length,
                        std,
                        init_sol_type,
                    )
                ] = fidelity_scores

                min_fidelity = min(fidelity_scores.values())

                if min_fidelity > best_min_fidelity:
                    best_min_fidelity = min_fidelity
                    optimal_control_pulse_sequences = optimal_pulse_sequences

rows = []

for hyperparameters, fidelity_scores in hyperparameters_to_scores.items():
    (
        num_iters,
        sequence_length,
        std,
        init_sol_type,
    ) = hyperparameters

    min_fidelity = min(fidelity_scores.values())

    row_dict = {
        "Sequence Length": sequence_length,
        "Num Iters": num_iters,
        "Std": std,
        "Init Sol Type": init_sol_type,
        "Min Fidelity": min_fidelity,
        "I": fidelity_scores["I"],
        "X": fidelity_scores["X"],
        "Y": fidelity_scores["Y"],
        "Z": fidelity_scores["Z"],
        "H": fidelity_scores["H"],
        "R_X_PI/4": fidelity_scores["R_X_PI/4"],
    }

    rows.append(row_dict)

df = pd.DataFrame(rows)
df = df.sort_values(by="Min Fidelity", ascending=False)
df.to_csv(f"hyperparameter_tuning_results_hc_round_2.csv", index=False)

torch.save(
    optimal_control_pulse_sequences,
    f"{best_min_fidelity}_optimal_control_pulses_hc_round_2.pt",
)
