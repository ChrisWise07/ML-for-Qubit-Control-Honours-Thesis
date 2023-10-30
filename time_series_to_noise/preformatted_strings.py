epoch_header = (
    "\n---------------- Epoch {epoch}/{num_epochs} ----------------\n"
)

training_header = "---------------- Training ----------------"

training_loss_progress = (
    "Epoch: {epoch}/{num_epochs}, Step {step}/{num_train_iterations}, "
    + "Loss: {loss:.4f}\n"
)

validation_header = "---------------- Validation ----------------"

validation_avg_loss_progress = (
    "Epoch: {epoch}/{num_epochs}, "
    + "Average Validation Loss: {avg_val_loss:.4f}\n"
)

testing_header = "\n---------------- Testing ----------------"

testing_avg_metrics = (
    "Average Parameter Score: {avg_parameter_score:.4f}, "
    + "\nAverage VO Fidelity: {avg_vo_fidelity:.4f}, "
    + "\nAverage Expectation Score: {avg_expectation_score:.4f}, "
    + "\nSum of Metrics: {sum_metric}, "
    + "\nAverage of Metrics: {avg_metric:.4f}, "
)
