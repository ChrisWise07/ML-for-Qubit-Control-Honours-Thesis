import torch
import pandas as pd
from control_pulse_optimisation.de_optimisation_of_pulses import (
    find_optimal_control_pulses,
)

torch.manual_seed(0)

hyperparameters_to_scores = {}
optimal_control_pulse_sequences = None

best_min_fidelity = 0
hyperparameter_data = []

for sequence_length in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    for population_size in [20]:
        for num_generations in [25]:
            for crossover_rate in [0.6, 0.8, 1.0]:
                for differential_weight in [0.5, 1.0, 2.0]:
                    (
                        fidelity_scores,
                        optimal_pulse_sequences,
                    ) = find_optimal_control_pulses(
                        sequence_length=sequence_length,
                        population_size=population_size,
                        num_generations=num_generations,
                        crossover_rate=crossover_rate,
                        differential_weight=differential_weight,
                    )

                    min_fidelity = min(fidelity_scores.values())

                    if min_fidelity > best_min_fidelity:
                        best_min_fidelity = min_fidelity
                        optimal_control_pulse_sequences = (
                            optimal_pulse_sequences
                        )

                    hyperparameter_scores = {
                        "Sequence Length": sequence_length,
                        "Population Size": population_size,
                        "Num Generations": num_generations,
                        "Crossover Rate": crossover_rate,
                        "Differential Weight": differential_weight,
                        "Min Fidelity": min_fidelity,
                        "I": fidelity_scores["I"],
                        "X": fidelity_scores["X"],
                        "Y": fidelity_scores["Y"],
                        "Z": fidelity_scores["Z"],
                        "H": fidelity_scores["H"],
                        "R_X_PI/4": fidelity_scores["R_X_PI/4"],
                    }

                    hyperparameter_data.append(hyperparameter_scores)

                    df = pd.DataFrame(hyperparameter_data)

                    df.to_csv(
                        f"new_hyperparameter_tuning_results_de.csv",
                        index=False,
                    )
