import argparse
import torch
import warnings
import os
import wandb
from datetime import datetime

from typing import Tuple

from time_series_to_noise import (
    EncoderWithMLP,
    SimpleANN,
    RNNWithMLP,
    ParameterLossTrainingWrapper,
    TraceDistanceLossWrapper,
    ExpectationLossWrapper,
    CombinedLossWrapper,
    CombinedDataset,
    ParamsDataset,
    VoDataset,
    ExpectationDataset,
    DEVICE,
    GenericTrainingWrapper,
    ParameterBasedLoss,
    TraceBasedLoss,
    ExpectationBasedLoss,
    CombinedLoss,
    BaseDataset,
)

from wandb_api_key import wandb_api_key

import datetime

from typing import Union

torch.manual_seed(0)

parser = argparse.ArgumentParser(
    description="Experiment hyperparameters & settings"
)

parser.add_argument(
    "--path_to_dataset",
    type=str,
    default="./QData_pickle/G_1q_XY_XZ_N3N6_comb",
    help="Path to the dataset",
)
parser.add_argument(
    "--offline",
    type=str,
    default="True",
    help="Whether to run the experiment offline and not log live to WandB "
    + "(default: False)",
)
parser.add_argument(
    "--num_params_to_predict",
    type=int,
    default=9,
    help="Number of parameters to predict (default: 9)",
    choices=[9, 12],
)
parser.add_argument(
    "--model_name",
    type=str,
    default="encoder_with_MLP",
    help="Name of the model to train (default: encoder_with_MLP)",
    choices=["encoder_with_MLP", "simple_ANN", "rnn_with_MLP"],
)
parser.add_argument(
    "--loss_name",
    type=str,
    default="trace_distance_loss",
    help="Name of the loss to train (default: trace_distance_loss)",
    choices=[
        "parameter_loss_mod",
        "trace_distance_loss",
        "expectation_loss",
        "combined_loss_mod",
    ],
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of epochs to train the model (default: 5)",
)
parser.add_argument(
    "--num_examples",
    type=int,
    default=1000,
    help="Number of examples to train the model (default: 1000)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training the model (default: 128)",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for training the model (default: 0.001)",
)
parser.add_argument(
    "--validation_split",
    type=float,
    default=0.2,
    help="Validation split for training the model (default: 0.2)",
    choices=range(0, 1),
    metavar="{0..1}",
)
parser.add_argument(
    "--test_split",
    type=float,
    default=0.1,
    help="Test split for training the model (default: 0.1)",
    choices=range(0, 1),
    metavar="{0..1}",
)
parser.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help=(
        "How often to print the training progress in terms of iterations (default: 5)"
    ),
    choices=range(1, 20000),
    metavar="{1..20000}",
)
parser.add_argument(
    "--n_heads",
    type=int,
    default=4,
    help="Number of heads for training the model (default: 4)",
)
parser.add_argument(
    "--dropout_encoder",
    type=float,
    default=0.1,
    help="Dropout for training the model (default: 0.2)",
    choices=range(0, 1),
    metavar="{0..1}",
)
parser.add_argument(
    "--n_encoder_layers",
    type=int,
    default=2,
    help="Number of encoder layers for training the model (default: 2)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Weight decay for training the model (default: 0.01)",
)
parser.add_argument(
    "--learning_rate_scheduler",
    type=str,
    default="None",
    help="Learning rate scheduler for training the model (default: None)",
    choices=["None", "cosine", "cosine_warm_restarts", "step"],
)
parser.add_argument(
    "--fixed_test_size",
    type=int,
    default=None,
    help="Fixed test size for training the model (default: None)",
)
parser.add_argument(
    "--test_on_validation_set",
    type=str,
    default="True",
    help="Whether to test on the validation set (default: True)",
    choices=["True", "False"],
)
parser.add_argument(
    "--use_activation",
    type=str,
    default="True",
    help=(
        "Whether to use the custom activation function"
        + " or use no activation function (default: True)"
    ),
)
parser.add_argument(
    "--input_type",
    type=str,
    default="pulse_parameters",
    help=(
        "Whether to use the time series or the pulse parameters as"
        + " input to the model (default: time_series)"
    ),
    choices=["time_series", "pulse_parameters"],
)


def check_train_val_test_split(
    validation_split: float, test_split: float
) -> None:
    """
    Check the train, validation, and test splits.

    Args:
        args: The arguments passed to the program.
    """
    if not (validation_split + test_split < 1):
        raise ValueError(
            "The train, validation, and test splits should sum up to 1."
        )

    if validation_split + test_split == 1:
        warnings.warn(
            "Validation and test split add up to 1. No training data will be used."
        )

    if test_split > validation_split:
        warnings.warn(
            "It is recommended that the validation split is larger than the test split."
        )

    if (validation_split + test_split) > 0.5:
        warnings.warn(
            "It is recommended that the validation and test split add up to less than 0.5."
        )


def check_args(args: argparse.Namespace) -> None:
    """
    Check the arguments passed to the program.

    Args:
        args: The arguments passed to the program.
    """
    check_train_val_test_split(args.validation_split, args.test_split)


args = parser.parse_args()


check_args(args)


def print_device_info() -> None:
    """
    Print the device info.
    """
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Current CUDA device:", torch.cuda.current_device())
        print("Number of GPUs:", torch.cuda.device_count())


def select_model(
    args: argparse.Namespace,
) -> Union[SimpleANN, EncoderWithMLP, RNNWithMLP]:
    """
    Select the model to train.

    Args:
        model_name: The name of the model to train.

    Returns:
        The model to train.
    """
    seq_dim = 6 if args.input_type == "pulse_parameters" else 2
    seq_len = 5 if args.input_type == "pulse_parameters" else 1024

    if args.model_name == "simple_ANN":
        return SimpleANN(
            seq_len=seq_len,
            seq_dim=seq_dim,
            use_activation=args.use_activation == "True",
            num_params_to_predict=args.num_params_to_predict,
        ).to(DEVICE)
    if args.model_name == "encoder_with_MLP":
        return EncoderWithMLP(
            n_heads=args.n_heads,
            dropout_encoder=args.dropout_encoder,
            n_encoder_layers=args.n_encoder_layers,
            d_model=32,
            dim_feedforward_encoder=64,
            use_activation=args.use_activation == "True",
            seq_dim=seq_dim,
            seq_len=seq_len,
            num_params_to_predict=args.num_params_to_predict,
        ).to(DEVICE)
    if args.model_name == "rnn_with_MLP":
        return RNNWithMLP(
            use_activation=args.use_activation == "True",
            num_params_to_predict=args.num_params_to_predict,
            seq_dim=seq_dim,
        ).to(DEVICE)
    raise ValueError(f"Unknown model name: {args.model_name}")


def select_learning_rate_scheduler(
    optimiser: torch.optim.Optimizer, scheduler_arg: str, num_epochs: int
) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_arg == "None":
        return None
    if scheduler_arg == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=num_epochs
        )
    if scheduler_arg == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=num_epochs // 5
        )
    if scheduler_arg == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=2, gamma=0.95
        )
    raise ValueError(f"Unknown learning rate scheduler: {scheduler_arg}")


def return_training_wrapper_and_dataset(
    loss_name: str,
) -> Tuple[
    GenericTrainingWrapper,
    BaseDataset,
    torch.nn.Module,
]:
    """
    Return the training wrapper and dataset for the given loss name.

    Args:
        loss_name: The name of the loss to train with.

    Returns:
        The training wrapper, dataset, and loss function corresponding 
        to the given loss name.
    """
    if loss_name == "parameter_loss_mod":
        return ParameterLossTrainingWrapper, ParamsDataset, ParameterBasedLoss
    if loss_name == "trace_distance_loss":
        return TraceDistanceLossWrapper, VoDataset, TraceBasedLoss
    if loss_name == "expectation_loss":
        return ExpectationLossWrapper, ExpectationDataset, ExpectationBasedLoss
    if loss_name == "combined_loss_mod":
        return CombinedLossWrapper, CombinedDataset, CombinedLoss
    raise ValueError(f"Unknown loss name: {loss_name}")


def main(args: argparse.Namespace) -> None:
    """
    Main function for the experiment.

    Args:
        args: The arguments passed to the program.
    """
    print_device_info()

    hearbeat = datetime.datetime.now().strftime("%m-%d-%H")

    experiment_name = (
        f"{hearbeat}"
        + f"-model:{args.model_name}"
        + f"-loss:{args.loss_name}"
        + f"-input_type:{args.input_type}"
        + f"-num_params:{args.num_params_to_predict}"
        + f"-use_activation:{args.use_activation}"
        + f"-lr:{args.learning_rate}"
        + f"-wd:{args.weight_decay}"
    )

    model_save_path = os.path.join(
        os.getcwd(), "models", f"{experiment_name}.pt"
    )

    os.environ["WANDB_API_KEY"] = wandb_api_key

    if args.offline == "True":
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        name=experiment_name,
        project="honours-research",
        config=args,
    )

    model = select_model(args)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    learning_rate_scheduler = select_learning_rate_scheduler(
        optimiser, args.learning_rate_scheduler, args.num_epochs
    )

    training_wrapper_obj, dataset_obj, criterion_obj = return_training_wrapper_and_dataset(
        args.loss_name
    )

    dataset = dataset_obj(
        path_to_dataset=args.path_to_dataset,
        num_examples=args.num_examples,
        need_delta=args.num_params_to_predict == 12,
        use_timeseries=args.input_type == "time_series",
    )

    criterion = criterion_obj(
        num_params_to_predict=args.num_params_to_predict
    ).to(DEVICE)

    training_wrapper_obj(
        model=model,
        optimiser=optimiser,
        dataset=dataset,
        criterion=criterion,
        learning_rate_scheduler=learning_rate_scheduler,
        num_epochs=args.num_epochs,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        print_freq=args.print_freq,
        validation_split=args.validation_split,
        test_split=args.test_split,
        model_save_path=model_save_path,
        test_on_validation_set=args.test_on_validation_set == "True",
        fixed_test_size=args.fixed_test_size,
        use_timeseries=args.input_type == "time_series",
    ).run()


if __name__ == "__main__":
    main(args=args)
