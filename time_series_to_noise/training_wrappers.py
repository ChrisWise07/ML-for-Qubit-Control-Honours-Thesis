import torch
from torch.utils.data import DataLoader, random_split
import wandb
from typing import List, Tuple, Union
import os
from abc import ABC, abstractmethod
from torchmetrics.functional.regression import (
    symmetric_mean_absolute_percentage_error,
)
from dataclasses import dataclass, InitVar, field


from time_series_to_noise.custom_dataset_objects import (
    BaseDataset,
    Mode,
)

from time_series_to_noise.preformatted_strings import (
    epoch_header,
    training_header,
    training_loss_progress,
    validation_header,
    testing_header,
    validation_avg_loss_progress,
    testing_avg_metrics,
)

from time_series_to_noise.custom_loss_functions import (
    TraceBasedLoss,
    ExpectationBasedLoss,
    ParameterBasedLoss,
    CombinedLoss,
)

from time_series_to_noise.utils import (
    return_estimated_VO_unital_for_batch,
    compute_mean_distance_matrix_for_batch_of_VO_matrices,
    fidelity_batch_of_matrices,
    calculate_expectation_values,
)

from time_series_to_noise.constants import DEVICE

torch.manual_seed(0)


@dataclass
class GenericTrainingWrapper(ABC):
    """Generic training wrapper for a model.

    Attributes:
        model (torch.nn.Module): The model to train..
        optimiser (torch.optim): The optimiser to use.
        dataset (BaseDataset): The dataset to use.
        learning_rate_scheduler (torch.optim.lr_scheduler):
            The learning rate scheduler to use.
        num_epochs (int): The number of epochs to train for.
        num_examples (int): The number of examples in the dataset.
        batch_size (int): The batch size.
        print_freq (int): The print frequency every n iterations.
        validation_split (float): The validation split.
        test_split (float): The test split.
        model_save_path (str): The path to save the model.
        test_on_validation_set (bool):
            Whether to test on the validation set.
        fixed_test_size (int):
            The fixed test size. Note this is only used in the
            __post_init__ method.
        use_timeseries:
            Whether to use the time series data or not. Note this is
            only used in the __post_init__ method.
    """

    model: torch.nn.Module
    optimiser: torch.optim
    dataset: BaseDataset
    criterion: torch.nn.Module
    learning_rate_scheduler: torch.optim.lr_scheduler
    num_epochs: int
    num_examples: int
    batch_size: int
    print_freq: int
    validation_split: float
    test_split: float
    model_save_path: str
    test_on_validation_set: bool
    use_timeseries: InitVar[bool]
    fixed_test_size: InitVar[int] = None

    def __post_init__(
        self, use_timeseries: bool, fixed_test_size: int = None
    ) -> None:
        if fixed_test_size is not None:
            print("Using fixed test size")
            self.test_size = fixed_test_size
            remaining_examples = len(self.dataset) - self.test_size
            self.train_size = int(
                (1 - self.validation_split) * remaining_examples
            )
            self.val_size = remaining_examples - self.train_size
            print(f"The train size is {self.train_size}")
            print(f"The val size is {self.val_size}")
            print(f"The test size is {self.test_size}")
        else:
            self.train_size = int(
                (1 - self.validation_split - self.test_split)
                * len(self.dataset)
            )
            self.val_size = int(self.validation_split * len(self.dataset))
            self.test_size = (
                len(self.dataset) - self.train_size - self.val_size
            )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.num_train_iterations = len(self.train_loader)

        self.should_update_scheduler_every_step = isinstance(
            self.optimiser,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        )

        seq_len = 1024 if use_timeseries else 5
        seq_dim = 2 if use_timeseries else 6
        self.pulses_shape = (self.batch_size, seq_len, seq_dim)
        self.pulses_alt_shape = (
            (self.num_examples % self.batch_size),
            seq_len,
            seq_dim,
        )

    def _return_loss_for_model_prediction(
        self,
        inputs: torch.Tensor,
        labels: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        """Return the loss for a model prediction.

        Args:
            inputs: The inputs to the model.
            labels: The labels for the model.

        Returns:
            The loss for the model prediction.
        """
        y_pred_parameters = self.model(inputs)
        return self.criterion(y_pred_parameters, *labels)

    def train(self, epoch: int):
        """Train the model for one epoch.

        Args:
            epoch: The epoch number.
        """

        print(epoch_header.format(epoch=epoch + 1, num_epochs=self.num_epochs))
        print(training_header)

        train_loss_sum = 0.0

        self.dataset.set_mode(Mode.TRAIN)
        self.model.train()
        with torch.enable_grad():
            for i, args in enumerate(self.train_loader):
                self.optimiser.zero_grad()
                args = [arg.to(DEVICE) for arg in args]

                loss = self._return_loss_for_model_prediction(
                    inputs=args[0], labels=args[1:]
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Loss is NaN or inf. Details: "
                        + f"\nLoss: {loss}"
                        + f"\nDoes model input have nan? {torch.isnan(args[0]).any()}"
                        + f"\nDoes model input have inf? {torch.isinf(args[0]).any()}"
                        + f"\nDoes model labels have nan? {any(torch.isnan(arg).any() for arg in args[1:])}"
                        + f"\nDoes model labels have inf? {any(torch.isinf(arg).any() for arg in args[1:])}"
                        + f"\nDoes model predictions have nan? {torch.isnan(self.model(args[0])).any()}"
                        + f"\nDoes model predictions have inf? {torch.isinf(self.model(args[0])).any()}"
                        + f"\nDoes model parameters have nan? {any(torch.isnan(p).any() for p in self.model.parameters())}"
                        + f"\nDoes model parameters have inf? {any(torch.isinf(p).any() for p in self.model.parameters())}"
                    )
                    raise ValueError("Loss is NaN or inf.")

                loss_for_logs = loss.item()

                train_loss_sum += loss_for_logs

                if (i + 1) % self.print_freq == 0:
                    print(
                        training_loss_progress.format(
                            epoch=epoch + 1,
                            num_epochs=self.num_epochs,
                            step=i + 1,
                            num_train_iterations=self.num_train_iterations,
                            loss=loss_for_logs,
                        )
                    )

                loss.backward()

                self.optimiser.step()
                if (
                    self.should_update_scheduler_every_step
                    and self.learning_rate_scheduler is not None
                ):
                    self.learning_rate_scheduler.step(
                        epoch + i / self.num_train_iterations
                    )

        if (
            not self.should_update_scheduler_every_step
            and self.learning_rate_scheduler is not None
        ):
            self.learning_rate_scheduler.step()

        avg_train_loss = train_loss_sum / (i + 1)

        wandb.log(
            {
                "training_loss": avg_train_loss,
            },
            step=epoch + 1,
        )

    def validate(self, epoch: int):
        """Validate the model for one epoch.

        Args:
            epoch: The epoch number.
        """
        print(validation_header)
        val_loss_sum = 0.0

        self.model.eval()
        self.dataset.set_mode(Mode.VAL)
        with torch.no_grad():
            for i, args in enumerate(self.val_loader):
                args = [arg.to(DEVICE) for arg in args]
                loss = self._return_loss_for_model_prediction(
                    inputs=args[0], labels=args[1:]
                )
                val_loss_sum += loss.item()

            avg_val_loss = val_loss_sum / (i + 1)

            print(
                validation_avg_loss_progress.format(
                    epoch=epoch + 1,
                    num_epochs=self.num_epochs,
                    avg_val_loss=avg_val_loss,
                )
            )

            wandb.log(
                {
                    "validation_loss": avg_val_loss,
                },
                step=epoch + 1,
            )

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model()

    def perform_epochs(self):
        """
        Perform the epochs, where each epoch consists of training and
        validation.

        """
        self.best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            self.train(epoch)
            self.validate(epoch)

    def test(self):
        """Test the model."""
        print(testing_header)

        data_loader = (
            self.val_loader
            if self.test_on_validation_set
            else self.test_loader
        )

        print(
            "Using validation data loader for testing."
        ) if self.test_on_validation_set else print(
            "Using test data loader for testing."
        )

        self.dataset.set_mode(Mode.TEST)

        length_of_data_loader = len(data_loader)

        parameter_smape_scores = torch.zeros(
            size=(length_of_data_loader,), dtype=torch.float32
        )

        vo_fidelities = torch.zeros(
            size=(length_of_data_loader,), dtype=torch.float32
        )

        expectations_smape_scores = torch.zeros(
            size=(length_of_data_loader,), dtype=torch.float32
        )

        expectations_mse_scores = torch.zeros(
            size=(length_of_data_loader,), dtype=torch.float32
        )

        with torch.no_grad():
            for i, args in enumerate(data_loader):
                y_pred_parameters = self.model(args["pulse_data"].to(DEVICE))

                parameter_smape_scores[
                    i
                ] = symmetric_mean_absolute_percentage_error(
                    y_pred_parameters, args["parameters"].to(DEVICE)
                )

                gt_Vx = args["Vx"].to(DEVICE)
                gt_Vy = args["Vy"].to(DEVICE)
                gt_Vz = args["Vz"].to(DEVICE)

                (
                    estimated_Vx,
                    estimated_Vy,
                    estimated_Vz,
                ) = return_estimated_VO_unital_for_batch(
                    batch_parameters=y_pred_parameters,
                )

                vo_fidelities[
                    i
                ] = compute_mean_distance_matrix_for_batch_of_VO_matrices(
                    estimated_VX=estimated_Vx,
                    estimated_VY=estimated_Vy,
                    estimated_VZ=estimated_Vz,
                    VX_true=gt_Vx,
                    VY_true=gt_Vy,
                    VZ_true=gt_Vz,
                    distance_measure=fidelity_batch_of_matrices,
                )

                predicted_expectation_values = calculate_expectation_values(
                    Vo_operators=torch.stack(
                        [estimated_Vx, estimated_Vy, estimated_Vz], dim=0
                    ),
                    control_unitaries=args["control_unitaries"].to(DEVICE),
                )

                ground_turth_expecations = args["expectations"].to(DEVICE)

                expectations_smape_scores[
                    i
                ] = symmetric_mean_absolute_percentage_error(
                    predicted_expectation_values,
                    ground_turth_expecations,
                )

                expectations_mse_scores[i] = torch.mean(
                    torch.square(
                        predicted_expectation_values - ground_turth_expecations
                    )
                )

            avg_parameter_score = (
                1 - 0.5 * torch.mean(parameter_smape_scores).item()
            )

            avg_vo_fidelity = torch.mean(vo_fidelities).item()

            avg_expectation_score = (
                1 - 0.5 * torch.mean(expectations_smape_scores).item()
            )

            avg_expectations_mse = torch.mean(expectations_mse_scores).item()

            sum_of_metrics = (
                avg_parameter_score + avg_vo_fidelity + avg_expectation_score
            )

            avg_of_metric = sum_of_metrics / 3

            print(
                testing_avg_metrics.format(
                    avg_parameter_score=avg_parameter_score,
                    avg_vo_fidelity=avg_vo_fidelity,
                    avg_expectation_score=avg_expectation_score,
                    sum_metric=sum_of_metrics,
                    avg_metric=avg_of_metric,
                )
            )

            wandb.log(
                {
                    "Parameter Score": avg_parameter_score,
                    "VO Fidelity": avg_vo_fidelity,
                    "Expectation Score": avg_expectation_score,
                    "Sum of Metrics": sum_of_metrics,
                    "Average of Metrics": avg_of_metric,
                    "Expectations MSE": avg_expectations_mse,
                }
            )

    def perform_shape_checks(
        self,
        tensor_names: List[str],
        tensors: List[torch.Tensor],
        expected_shapes: List[Tuple[int]],
    ):
        """Perform shape checks on the tensors.

        Args:
            tensor_names: The names of the tensors.
            tensors: The tensors.
            expected_shapes: The expected shapes of the tensors.
        """
        for name, tensor, expected_shapes in zip(
            tensor_names, tensors, expected_shapes
        ):
            if tensor.shape not in expected_shapes:
                raise ValueError(
                    f"{name} should have shape {expected_shapes[0]} or "
                    + f"{expected_shapes[1]} but has shape {tensor.shape}"
                )

    def run(self):
        """Run the training, validation, and testing."""
        self.perform_epochs()

        self.load_best_model()

        self.test()

    def save_model(self) -> None:
        """Save the model."""
        torch.save(self.model.state_dict(), self.model_save_path)

    def load_best_model(self):
        """Load the best model."""
        state_dict = torch.load(self.model_save_path)
        self.model.load_state_dict(state_dict)


@dataclass
class ParameterLossTrainingWrapper(GenericTrainingWrapper):
    """A training wrapper for a model with a parameter loss."""

    def __post_init__(
        self, use_timeseries: bool, fixed_test_size: int = None
    ) -> None:
        super().__post_init__(use_timeseries, fixed_test_size)
        self.loss_name = "parameter_loss"

        num_params_to_predict = (
            self.dataset.get_number_of_parameters_being_predicted()
        )

        params_shape = (self.batch_size, num_params_to_predict)

        alt_params_shape = (
            (self.num_examples % self.batch_size),
            num_params_to_predict,
        )

        _, (pulses, gt_params) = next(enumerate(self.train_loader))

        self.perform_shape_checks(
            tensor_names=["pulse_data", "gt_params"],
            tensors=[pulses, gt_params],
            expected_shapes=[
                [self.pulses_shape, self.pulses_alt_shape],
                [params_shape, alt_params_shape],
            ],
        )


@dataclass
class TraceDistanceLossWrapper(GenericTrainingWrapper):
    def __post_init__(
        self, use_timeseries: bool, fixed_test_size: int = None
    ) -> None:
        super().__post_init__(use_timeseries, fixed_test_size)
        self.loss_name = "trace_distance_loss"
        self.vo_shape = (self.batch_size, 2, 2)
        self.alt_vo_shape = ((self.num_examples % self.batch_size), 2, 2)
        _, (pulses, Vx, Vy, Vz) = next(enumerate(self.train_loader))

        self.perform_shape_checks(
            tensor_names=["pulse_data", "VX_true", "VY_true", "VZ_true"],
            tensors=[
                pulses,
                Vx,
                Vy,
                Vz,
            ],
            expected_shapes=[
                [self.pulses_shape, self.pulses_alt_shape],
                [self.vo_shape, self.alt_vo_shape],
                [self.vo_shape, self.alt_vo_shape],
                [self.vo_shape, self.alt_vo_shape],
            ],
        )


@dataclass
class ExpectationLossWrapper(GenericTrainingWrapper):
    def __post_init__(
        self, use_timeseries: bool, fixed_test_size: int = None
    ) -> None:
        super().__post_init__(use_timeseries, fixed_test_size)
        self.loss_name = "expectation_loss"
        self.expectation_shape = (self.batch_size, 18)
        self.control_unitary_shape = (self.batch_size, 2, 2)

        self.alt_expectation_shape = (
            (self.num_examples % self.batch_size),
            18,
        )

        self.alt_control_unitary_shape = (
            (self.num_examples % self.batch_size),
            2,
            2,
        )

        _, (pulses, expectations, control_unitaries) = next(
            enumerate(self.train_loader)
        )

        self.perform_shape_checks(
            tensor_names=["pulse_data", "expectations", "control_unitaries"],
            tensors=[pulses, expectations, control_unitaries],
            expected_shapes=[
                [self.pulses_shape, self.pulses_alt_shape],
                [self.expectation_shape, self.alt_expectation_shape],
                [self.control_unitary_shape, self.alt_control_unitary_shape],
            ],
        )


@dataclass
class CombinedLossWrapper(GenericTrainingWrapper):
    def __post_init__(
        self, use_timeseries: bool, fixed_test_size: int = None
    ) -> None:
        super().__post_init__(use_timeseries, fixed_test_size)
        self.loss_name = "combined_loss"
