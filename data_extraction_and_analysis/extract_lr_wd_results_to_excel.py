import pandas as pd
import wandb

api = wandb.Api()

runs = api.runs(
    "chriswise/Honours-Project-Phase-2-LR-WD-Results"
)

data, names = [], []
for run in runs:
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}

    config_mapped = {
        "Model": config.get("model_name"),
        "Learning Rate": config.get("learning_rate"),
        "Weight Decay": config.get("weight_decay"),
    }

    summary = run.summary._json_dict

    summary_mapped = {
        "Average of Metrics": summary.get("Average of Metrics"),
        "Expectation MSE": summary.get("Expectations MSE"),
    }

    merged = {**config_mapped, **summary_mapped}
    data.append(merged)

df = pd.DataFrame(data)

ordering = {
    "Model": ["simple_ANN", "rnn_with_MLP", "encoder_with_MLP"],
    "Learning Rate": [0.0001, 0.001, 0.01],
    "Weight Decay": [0.001, 0.01, 0.1],
}

for col, order in ordering.items():
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)

df.sort_values(by=list(ordering.keys()), inplace=True)

df.to_excel("learning_rate_weight_decay_results.xlsx")
