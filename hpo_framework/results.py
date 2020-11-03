import pandas as pd


class TuningResult:
    def __init__(self, evaluation_ids: list, timestamps: list, losses: list, configurations: tuple, best_val_loss: float,
                 best_configuration: dict, wall_clock_time, test_loss: float, successful=True, did_warmstart=False,
                 budget=100.0):
        self.evaluation_ids = evaluation_ids  # list
        self.timestamps = timestamps  # list
        self.losses = losses  # list
        self.configurations = configurations  # tuple of dictionaries
        self.best_val_loss = best_val_loss  # float
        self.best_configuration = best_configuration  # dictionary
        self.wall_clock_time = wall_clock_time
        self.successful = successful  # Flag that indicates, whether the run was finished successfully or not
        self.test_loss = test_loss  # Loss on test set for best found HP configuration
        self.did_warmstart = did_warmstart  # Flag that indicates, whether a warmstart took place
        self.budget = budget  # Used optimization budget [%] (e.g. BOHB varies the budget in each iteration)


class TrialResult:
    def __init__(self, trial_result_df: pd.DataFrame, best_trial_configuration: dict, best_val_loss: float,
                 best_test_loss: float, hpo_library: str, hpo_method: str, did_warmstart: bool):
        self.trial_result_df = trial_result_df
        self.best_trial_configuration = best_trial_configuration
        self.best_val_loss = best_val_loss
        self.best_test_loss = best_test_loss
        self.hpo_library = hpo_library
        self.hpo_method = hpo_method
        self.did_warmstart = did_warmstart


class MetricsResult:
    def __init__(self, wall_clock_time, time_outperform_default, area_under_curve, mean_test_loss, loss_ratio,
                 interquantile_range, time_best_config, evals_for_best_config, number_of_crashes):
        self.wall_clock_time = wall_clock_time
        self.time_outperform_default = time_outperform_default
        self.area_under_curve = area_under_curve
        self.mean_test_loss = mean_test_loss
        self.loss_ratio = loss_ratio
        self.interquantile_range = interquantile_range
        self.time_best_config = time_best_config
        self.evals_for_best_config = evals_for_best_config
        self.number_of_crashes = number_of_crashes
