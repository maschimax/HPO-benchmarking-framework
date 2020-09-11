import preprocessing as pp
import time
import matplotlib.pyplot as plt

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
import hpbandster.visualization as hpvis

from workers import RandomForestWorker
from workers import SVMWorker

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

    # worker = SVMWorker(X_train, X_val, y_train, y_val,
    #                    nameserver='127.0.0.1', run_id='example1')
    # worker.run(background=True)

    # Step 3: Run an optimizer
    bohb = BOHB(configspace=worker.get_configspace(), run_id='example1',
                nameserver='127.0.0.1', min_budget=1, max_budget=9, eta=3.0)
    res = bohb.run(n_iterations=10)

    # Step 4: Shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Step 5: Analysis
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    trajectory = res.get_incumbent_trajectory()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('Best loss: ', min(trajectory['losses']))
    print('A total of %i unique configurations where sampled.' %
          len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))

    # Plotting results
    # all_runs = res.get_all_runs()
    #
    # hpvis.losses_over_time(all_runs)
    #
    # plt.show()
