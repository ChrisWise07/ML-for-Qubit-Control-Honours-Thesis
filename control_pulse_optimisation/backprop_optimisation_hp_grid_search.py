import torch
import pandas as pd
from backprop_optimisation_of_pulses import optimise_control_pulse

torch.manual_seed(0)

hyperparameters_to_scores = {}

for sequence_length in [256, 512]:
    for noise_strength in [0.2, 0.4, 0.8, 1.6]:
        for max_amp in [50, 100, 200]:
            for objective_func_name in ["expectations", "vo_operators"]:
                fidelity_scores = optimise_control_pulse(
                    sequence_length=sequence_length,
                    noise_strength=noise_strength,
                    max_amp=max_amp,
                    objective_func_name=objective_func_name,
                )

                hyperparameters_to_scores[
                    (
                        sequence_length,
                        noise_strength,
                        max_amp,
                        objective_func_name,
                    )
                ] = fidelity_scores

rows = []

for hyperparameters, fidelity_scores in hyperparameters_to_scores.items():
    (
        sequence_length,
        noise_strength,
        max_amp,
        objective_func_name,
    ) = hyperparameters

    min_fidelity = min(fidelity_scores.values())

    row_dict = {
        "Sequence Length": sequence_length,
        "Noise Strength": noise_strength,
        "Max Amplitude": max_amp,
        "Objective Function": objective_func_name,
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
df.to_csv(f"backprop_through_sim_benchmarks_2.csv", index=False)
