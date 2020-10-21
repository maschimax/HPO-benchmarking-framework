from datasets.dummy import preprocessing as pp
import matplotlib.pyplot as plt
import os
from pathlib import Path

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch
import hpbandster.core.result as hpres

from hpo.outdated.workers import RandomForestWorker
from hpo.outdated.workers import SVMWorker
from hpo.outdated.workers import KerasRegressor

# ML-algorithm
ALGORITHM = 'RandomForestRegressor'  # 'SVR', 'RandomForestRegressor', 'Keras'
# HPO-method
OPTIMIZER = 'BOHB'  # 'BOHB', 'RandomSearch', 'Hyperband'

# Preprocessing
abs_folder_path = os.path.abspath(path='/home/max/Desktop/Projects/housing_regression/datasets')
data_folder = Path(abs_folder_path)
train_file = "train.csv"
test_file = "test.csv"
submission_file = "sample_submission.csv"

train_raw = pp.load_data(data_folder, train_file)
test_raw = pp.load_data(data_folder, test_file)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

# Step 1: Start a nameserver
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
if ALGORITHM == 'RandomForestRegressor':
    worker = RandomForestWorker(X_train, X_val, y_train, y_val,
                                nameserver='127.0.0.1', run_id='example1')
    worker.run(background=True)
    isKeras = False

elif ALGORITHM == 'SVR':
    worker = SVMWorker(X_train, X_val, y_train, y_val,
                       nameserver='127.0.0.1', run_id='example1')
    worker.run(background=True)
    isKeras = False

elif ALGORITHM == 'Keras':
    worker = KerasRegressor(X_train, X_val, y_train, y_val, nameserver='127.0.0.1', run_id='example1')
    worker.run(background=True)
    isKeras = True
else:
    isKeras = False

# Step 3: Run an optimizer
result_logger = hpres.json_result_logger(directory='.',
                                         overwrite=True)
if not isKeras:
    if OPTIMIZER == 'BOHB':
        optimizer = BOHB(configspace=worker.get_warmstart_configspace(), run_id='example1',
                         nameserver='127.0.0.1', min_budget=10, max_budget=10, eta=3.0,
                         result_logger=result_logger)
        res = optimizer.run(n_iterations=1)

    elif OPTIMIZER == 'RandomSearch':
        optimizer = RandomSearch(configspace=worker.get_configspace(), run_id='example1',
                                 nameserver='127.0.0.1', min_budget=1, max_budget=9, eta=3.0,
                                 result_logger=result_logger)
        res = optimizer.run(n_iterations=10)
else:
    if OPTIMIZER == 'BOHB':
        optimizer = BOHB(configspace=worker.get_configspace(), run_id='example1',
                         nameserver='127.0.0.1', min_budget=3, max_budget=100, eta=3.0,
                         result_logger=result_logger)
        res = optimizer.run(n_iterations=5)
    elif OPTIMIZER == 'RandomSearch':
        optimizer = RandomSearch(configspace=worker.get_configspace(), run_id='example1',
                                 nameserver='127.0.0.1', min_budget=3, max_budget=100, eta=3.0,
                                 result_logger=result_logger)
        res = optimizer.run(n_iterations=5)

# Step 4: Shutdown
optimizer.shutdown(shutdown_workers=False)

# Step 5: Analysis
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
trajectory = res.get_incumbent_trajectory()

####
# Reuse the warm start results
# NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
# NS.start()

prev_run = hpres.logged_results_to_HBS_result(directory='.')
bohb = BOHB(configspace=worker.get_configspace(), run_id='example1',
            nameserver='127.0.0.1', min_budget=1, max_budget=10, eta=3.0,
            result_logger=result_logger, previous_result=prev_run)
res2 = bohb.run(n_iterations=5)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()
####
all_runs = res2.get_all_runs()

id2config = res2.get_id2config_mapping()
incumbent = res2.get_incumbent_id()
trajectory = res2.get_incumbent_trajectory()

print('Best found configuration:', id2config[incumbent]['config'])
print('Best loss: ', min(trajectory['losses']))
print('A total of %i unique configurations where sampled.' %
      len(id2config.keys()))
print('A total of %i runs where executed.' % len(res2.get_all_runs()))

# Plot the learning curve
fig, ax = plt.subplots()
ax.plot(trajectory['losses'], trajectory['times_finished'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.show()
