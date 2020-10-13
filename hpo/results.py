import pandas as pd


class TuningResult:
    def __init__(self, evaluation_ids: list, timestamps: list, losses: list, configurations: tuple, best_loss: float,
                 best_configuration: dict, wall_clock_time, successful: True):
        self.evaluation_ids = evaluation_ids  # list
        self.timestamps = timestamps  # list
        self.losses = losses  # list
        self.configurations = configurations  # tuple of dictionaries
        self.best_loss = best_loss  # float
        self.best_configuration = best_configuration  # dictionary
        self.wall_clock_time = wall_clock_time
        self.successful = successful  # Flag that indicates, whether the run was finished successfully or not


class TrialResult:
    def __init__(self, trial_result_df: pd.DataFrame, best_trial_configuration: dict, best_trial_loss: float,
                 hpo_library: str, hpo_method: str):
        self.trial_result_df = trial_result_df
        self.best_trial_configuration = best_trial_configuration
        self.best_loss = best_trial_loss
        self.hpo_library = hpo_library
        self.hpo_method = hpo_method


class MetricsResult:
    def __init__(self, wall_clock_time, time_outperform_default, area_under_curve, best_mean_loss, loss_ratio, interquantile_range,
                 time_best_config, evals_for_best_config, number_of_crashes):
        self.wall_clock_time = wall_clock_time
        self.time_outperform_default = time_outperform_default
        self.area_under_curve = area_under_curve
        self.best_mean_loss = best_mean_loss
        self.loss_ratio = loss_ratio
        self.interquantile_range = interquantile_range
        self.time_best_config = time_best_config
        self.evals_for_best_config = evals_for_best_config
        self.number_of_crashes = number_of_crashes
