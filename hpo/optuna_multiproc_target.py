import optuna


def load_study_and_optimize(st_name, st_storage, n_func_evals, objective_func):
    this_study = optuna.load_study(st_name, st_storage)
    this_study.optimize(objective_func, n_func_evals)
    return

