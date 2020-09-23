import preprocessing as pp
import matplotlib.pyplot as plt

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

# loading data and preprocessing
FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
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
result_logger = hpres.json_result_logger(directory=r"C:\Users\Max\Documents\GitHub\housing_regression\logs",
                                         overwrite=True)
if not isKeras:
    if OPTIMIZER == 'BOHB':
        optimizer = BOHB(configspace=worker.get_configspace(), run_id='example1',
                         nameserver='127.0.0.1', min_budget=1, max_budget=9, eta=3.0,
                         result_logger=result_logger)
        res = optimizer.run(n_iterations=10)

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
optimizer.shutdown(shutdown_workers=True)
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

# Plot the learning curve
fig, ax = plt.subplots()
ax.plot(trajectory['losses'], trajectory['times_finished'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.show()
