import torch
import pandas as pd
from control_pulse_optimisation.ga_optimisation_of_pulses import (
    find_optimal_control_pulses,
)

torch.manual_seed(0)

hyperparameters_to_scores = {}
optimal_control_pulse_sequences = None

best_min_fidelity = 0
hyperparameter_data = []

for sequence_length in [512]:
    for std in [0.03125, 0.0625, 0.125]:
        for population_size in [10, 20, 40]:
            for num_generations in [6, 13, 25]:
                for crossover_rate in [0.2, 0.4, 0.8]:
                    for elitism_rate in [0.06, 0.12, 0.24]:
                        for init_sol_type in [
                            "noiseless_ideal",
                            "uniform_distro",
                            "normal_distro",
                        ]:
                            (
                                fidelity_scores,
                                optimal_pulse_sequences,
                            ) = find_optimal_control_pulses(
                                sequence_length=sequence_length,
                                std=std,
                                population_size=population_size,
                                num_generations=num_generations,
                                crossover_rate=crossover_rate,
                                elitism_rate=elitism_rate,
                                init_sol_type=init_sol_type,
                            )

                            min_fidelity = min(fidelity_scores.values())

                            if min_fidelity > best_min_fidelity:
                                best_min_fidelity = min_fidelity
                                optimal_control_pulse_sequences = (
                                    optimal_pulse_sequences
                                )

                            hyperparameter_scores = {
                                "Sequence Length": sequence_length,
                                "Std": std,
                                "Population Size": population_size,
                                "Num Generations": num_generations,
                                "Crossover Rate": crossover_rate,
                                "Elitism Rate": elitism_rate,
                                "Init Sol Type": init_sol_type,
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
                                f"hyperparameter_tuning_results_ga_4_32.csv",
                                index=False,
                            )
