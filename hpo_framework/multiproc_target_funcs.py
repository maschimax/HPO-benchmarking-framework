import optuna
from hpo_framework.hpbandster_worker import HPBandsterWorker
import os
from hyperopt import fmin


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


def initialize_worker(ml_algo, optimizer_obj, nameserver, run_id):
    """
    Function to target for a multiprocessing.Process in class HPBandsterOptimizer
    :param ml_algo: str
        The Machine Learning algorithm to be used
    :param optimizer_obj:
    :param nameserver: str
        Nameserver for the communication of the workers.
    :param run_id: str
        Identifier for this hpbandster-run.
    :return:
    """
    worker = HPBandsterWorker(ml_algorithm=ml_algo, optimizer_object=optimizer_obj,
                              nameserver=nameserver, run_id=run_id)
    worker.run(background=False)
    return


def hyperopt_target1(objective, hyperopt_space, trials, this_optimizer, n_func_evals, rand_num_generator):
    res = fmin(fn=objective, space=hyperopt_space, trials=trials, algo=this_optimizer,
               max_evals=n_func_evals, rstate=rand_num_generator)
    return


def hyperopt_target2():
    # check args in hyperopt.mongoexp main_worker()
    os.system("hyperopt-mongo-worker --mongo=localhost:27017/mongo_hpo --poll-interval=0.1 --reserve-timeout=20.0")

    return
