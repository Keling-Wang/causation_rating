from transformers import TrainerCallback
import optuna


# Custom pruning callback for Transformers
class TransformersPruningCallback(TrainerCallback):
    def __init__(self, trial, metric, prune_step, prune_threshold, step_offset=0):
        self.trial = trial
        self.metric = metric
        self.prune_step = prune_step
        self.prune_threshold = prune_threshold
        self.step_offset = step_offset  # Existing step_offset

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current_score = metrics.get(self.metric)
        if current_score is None:
            return

        # Adjust step with offset for reporting
        adjusted_step = state.global_step + self.step_offset

        # Use state.global_step for pruning checks (resets each fold)
        if self.prune_step is not None and state.global_step == self.prune_step:
            # Apply custom pruning rule at prune_step
            if current_score < self.prune_threshold:
                print(f"Pruning trial at adjusted step {adjusted_step} with {self.metric}={current_score:.4f}\n")
                self.trial.report(current_score, step=adjusted_step)
                #wandb.finish() # if using wandb please uncomment this line
                raise optuna.exceptions.TrialPruned()

        # If you have multiple prune steps per fold
        # Example for pruning at 2 * prune_step
        if self.prune_step is not None and state.global_step == int(self.prune_step * 2):
            # Apply custom pruning rule at 2 * prune_step
            if current_score < self.prune_threshold:
                print(f"Pruning trial at adjusted step {adjusted_step} with {self.metric}={current_score:.4f}")
                self.trial.report(current_score, step=adjusted_step)
                #wandb.finish() # if using wandb please uncomment this line
                raise optuna.exceptions.TrialPruned()

        # Report intermediate objective value to optuna using adjusted_step
        self.trial.report(current_score, step=adjusted_step)

        # Prune trial if needed
        if self.trial.should_prune():
            #wandb.finish() # if using wandb please uncomment this line
            raise optuna.exceptions.TrialPruned()