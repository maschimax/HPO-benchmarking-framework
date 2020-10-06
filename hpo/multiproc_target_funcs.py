import optuna
from hpo.hpbandster_worker import HPBandsterWorker


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


def initialize_worker(x_train, x_val, y_train, y_val, ml_algo, optimizer_obj, nameserver, run_id):
    """
    Function to target for a multiprocessing.Process in class HPBandsterOptimizer
    :param x_train: pd.DataFrame
        Training data.
    :param x_val: pd.DataFrame
        Validation data.
    :param y_train: pd.Series
        Training labels.
    :param y_val: pd.Series
        Validation labels.
    :param ml_algo: str
        The Machine Learning algorithm to be used
    :param optimizer_obj:
    :param nameserver: str
        Nameserver for the communication of the workers.
    :param run_id: str
        Identifier for this hpbandster-run.
    :return:
    """
    worker = HPBandsterWorker(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val,
                              ml_algorithm=ml_algo, optimizer_object=optimizer_obj,
                              nameserver=nameserver, run_id=run_id)
    worker.run(background=False)
    return


def test_func(i):
    print('Test: ', i)
    return
