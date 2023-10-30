import pandas as pd
import wandb

api = wandb.Api()

runs = api.runs(
    "chriswise/Honours-Research-Phase-2-Structural-Hyperparameters"
)

data, names = [], []
for run in runs:
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

    config_mapped = {
        "Loss Function": config.get("loss_name"),
        "Network": config.get("model_name"),
        "Input Type": config.get("input_type"),
        "Num of params to predict": config.get("num_params_to_predict"),
        "Uses Activation": config.get("use_activation"),
    }

    summary = run.summary._json_dict

    summary_mapped = {
        "Sum of Metrics": summary.get("Sum of Metrics"),
        "Average of Metrics": summary.get("Average of Metrics"),
        "Expectation MSE": summary.get("Expectations MSE"),
    }

    merged = {**config_mapped, **summary_mapped}
    data.append(merged)

df = pd.DataFrame(data)

ordering = {
    "Loss Function": [
        "parameter_loss",
        "trace_distance_loss",
        "expectation_loss",
        "combined_loss"
    ],
    "Network": ["simple_ANN", "rnn_with_MLP", "encoder_with_MLP"],
    "Input Type": ["pulse_parameters", "time_series"],
    "Num of params to predict": [9, 12],
    "Uses Activation": ["True", "False"],
}

for col, order in ordering.items():
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)

df.sort_values(by=list(ordering.keys()), inplace=True)

df.to_excel("strucutral_hp_results.xlsx")
