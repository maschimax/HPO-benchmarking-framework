import optuna


def load_study_and_optimize(st_name, st_storage, n_func_evals, objective_func):
    """
    Function to target for a multiprocessing.Process in class OptunaOptimizer
    :param st_name: str
        Name of the optuna study.
    :param st_storage: str
        Database URL.
    :param n_func_evals: int
        Number of function evaluations for this multiprocessing.Process
    :param objective_func:
        The objective function to be evaluated by the HPO-method
    :return:
    """
    this_study = optuna.load_study(st_name, st_storage)
    this_study.optimize(objective_func, n_func_evals)
    return

