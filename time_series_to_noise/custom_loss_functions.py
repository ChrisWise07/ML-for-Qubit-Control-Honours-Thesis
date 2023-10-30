import torch

from time_series_to_noise.utils import (
    return_estimated_VO_unital_for_batch,
    calculate_expectation_values,
    compute_mean_distance_matrix_for_batch_of_VO_matrices,
)

torch.manual_seed(0)


def normalise_expectation_values(expectations: torch.Tensor) -> torch.Tensor:
    """
    Normalise the expectation values to be between 0 and 1.

    Args:
        expectations (torch.Tensor): expectation values

    Returns:
        torch.Tensor: normalised expectation values
    """
    return (expectations + 1) / 2


class ExpectationBasedLoss(torch.nn.Module):
    """
    Calculates the MSE between the predicted and ground truth
    expectation values. Calculated for the whole batch.
    """

    def __init__(self, *args, **kwargs):
        super(ExpectationBasedLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(
        self,
        y_pred_parameters: torch.Tensor,
        expectations: torch.Tensor,
        control_unitaries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss between the estimated expectation values and
        the true expectation values.

        Args:
            y_pred_parameters (torch.Tensor): predicted parameters
            expectations (torch.Tensor): true expectation values
            control_unitaries (torch.Tensor): control unitaries

        Returns:
            torch.Tensor: loss
        """
        (
            estimated_Vx,
            estimated_Vy,
            estimated_Vz,
        ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

        predicted_expectation_values = calculate_expectation_values(
            estimated_Vx, estimated_Vy, estimated_Vz, control_unitaries
        )

        normalised_predicted_expectation_values = normalise_expectation_values(
            predicted_expectation_values
        )

        normalised_ground_truth_expectation_values = (
            normalise_expectation_values(expectations)
        )

        return self.bce_loss(
            normalised_predicted_expectation_values,
            normalised_ground_truth_expectation_values,
        )


class TraceBasedLoss(torch.nn.Module):
    """
    Calculate the squared l2 norm of the difference between the
    predicted and true noise matrices. Calculated for the whole batch.
    """

    def forward(
        self,
        y_pred_parameters: torch.Tensor,
        VX_true: torch.Tensor,
        VY_true: torch.Tensor,
        VZ_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss between the estimated noise matrices and the
        true noise matrices.

        Args:
            y_pred_parameters (torch.Tensor): predicted parameters
            VX_true (torch.Tensor): true noise encoding matrix V_X
            VY_true (torch.Tensor): true noise encoding matrix V_Y
            VZ_true (torch.Tensor): true noise encoding matrix V_Z

        Returns:
            torch.Tensor: loss
        """
        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

        return compute_mean_distance_matrix_for_batch_of_VO_matrices(
            estimated_VX,
            estimated_VY,
            estimated_VZ,
            VX_true,
            VY_true,
            VZ_true,
        )


class ParameterBasedLoss(torch.nn.Module):
    def __init__(self, num_params_to_predict=9):
        super(ParameterBasedLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()
        self.num_params_to_predict = num_params_to_predict
        if num_params_to_predict == 9:
            self.psi_indices = torch.tensor([0, 3, 6])
            self.theta_indices = torch.tensor([1, 4, 7])
            self.mu_indices = torch.tensor([2, 5, 8])
        elif num_params_to_predict == 12:
            self.psi_indices = torch.tensor([0, 4, 8])
            self.theta_indices = torch.tensor([1, 5, 9])
            self.delta_indices = torch.tensor([2, 6, 10])
            self.mu_indices = torch.tensor([3, 7, 11])

    def normalize_parameters(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.psi_indices] = (
            x[:, self.psi_indices] + torch.pi / 2
        ) / torch.pi
        x[:, self.theta_indices] = x[:, self.theta_indices] / (torch.pi / 2)

        if self.num_params_to_predict == 12:
            x[:, self.delta_indices] = (
                x[:, self.delta_indices] + torch.pi / 2
            ) / torch.pi

        return x

    def forward(
        self, pred_parameters: torch.Tensor, true_parameters: torch.Tensor
    ) -> torch.Tensor:
        norm_pred_parameters = self.normalize_parameters(pred_parameters)
        norm_true_parameters = self.normalize_parameters(true_parameters)

        return self.bce_loss(norm_pred_parameters, norm_true_parameters)


class CombinedLoss(torch.nn.Module):
    def __init__(self, num_params_to_predict=9):
        super(CombinedLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()
        self.num_params_to_predict = num_params_to_predict
        if num_params_to_predict == 9:
            self.psi_indices = torch.tensor([0, 3, 6])
            self.theta_indices = torch.tensor([1, 4, 7])
            self.mu_indices = torch.tensor([2, 5, 8])
        elif num_params_to_predict == 12:
            self.psi_indices = torch.tensor([0, 4, 8])
            self.theta_indices = torch.tensor([1, 5, 9])
            self.delta_indices = torch.tensor([2, 6, 10])
            self.mu_indices = torch.tensor([3, 7, 11])

    def normalize_parameters(self, x: torch.Tensor) -> torch.Tensor:
        x[:, self.psi_indices] = (
            x[:, self.psi_indices] + torch.pi / 2
        ) / torch.pi
        x[:, self.theta_indices] = x[:, self.theta_indices] / (torch.pi / 2)

        if self.num_params_to_predict == 12:
            x[:, self.delta_indices] = (
                x[:, self.delta_indices] + torch.pi / 2
            ) / torch.pi

        return x

    def forward(
        self,
        y_pred_parameters: torch.Tensor,
        true_parameters: torch.Tensor,
        VX_true: torch.Tensor,
        VY_true: torch.Tensor,
        VZ_true: torch.Tensor,
        true_expectations: torch.Tensor,
        control_unitaries: torch.Tensor,
    ) -> torch.Tensor:
        norm_pred_parameters = self.normalize_parameters(y_pred_parameters)
        norm_true_parameters = self.normalize_parameters(true_parameters)

        parameter_loss = self.bce_loss(
            norm_pred_parameters, norm_true_parameters
        )

        (
            estimated_VX,
            estimated_VY,
            estimated_VZ,
        ) = return_estimated_VO_unital_for_batch(y_pred_parameters)

        trace_loss = compute_mean_distance_matrix_for_batch_of_VO_matrices(
            estimated_VX,
            estimated_VY,
            estimated_VZ,
            VX_true,
            VY_true,
            VZ_true,
        )

        predicted_expectation_values = calculate_expectation_values(
            Vo_operators=torch.stack(
                (estimated_VX, estimated_VY, estimated_VZ), dim=0
            ),
            control_unitaries=control_unitaries,
        )

        normalised_predicted_expectation_values = normalise_expectation_values(
            predicted_expectation_values
        )

        normalised_ground_truth_expectation_values = (
            normalise_expectation_values(true_expectations)
        )

        expectation_loss = self.bce_loss(
            normalised_predicted_expectation_values,
            normalised_ground_truth_expectation_values,
        )

        return parameter_loss + trace_loss + expectation_loss
