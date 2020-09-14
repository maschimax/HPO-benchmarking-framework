import preprocessing as pp

import hpbandster.core.nameserver as hpns
import ConfigSpace as CS

from workers import RandomForestWorker

from hyperopt import fmin
from hyperopt import hp

from sklearn.metrics import mean_squared_error
from math import sqrt

# def objective(x):
#     return sqrt(mean_squared_error())

if __name__ == "__main__":

    # loading data and preprocessing
    FOLDER = 'datasets'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    SAMPLE_SUB = 'sample_submission.csv'

    train_raw = pp.load_data(FOLDER, TRAIN_FILE)
    test_raw = pp.load_data(FOLDER, TEST_FILE)
    X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                        count_encoding=False)

    # Step 1: Start a nameserver
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    worker = RandomForestWorker(X_train, X_val, y_train, y_val,
                                nameserver='127.0.0.1', run_id='example1')
    worker.run(background=True)
    isKeras = False

    #####################

    cs = worker.get_configspace()

    space = {}
    for h in cs.get_hyperparameters():
        if type(h) == CS.hyperparameters.OrdinalHyperparameter:
            space[h.name] = hp.quniform(h.name, 0, len(h.sequence) - 1, q=1)
        elif type(h) == CS.hyperparameters.CategoricalHyperparameter:
            space[h.name] = hp.choice(h.name, h.choices)
        elif type(h) == CS.hyperparameters.UniformIntegerHyperparameter:
            space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
        elif type(h) == CS.hyperparameters.UniformFloatHyperparameter:
            space[h.name] = hp.uniform(h.name, h.lower, h.upper)
