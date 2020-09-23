import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
import matplotlib.pyplot as plt

from hpo.outdated.workers import RandomForestWorker
import preprocessing as pp

numWorkers = 4

# loading data and preprocessing
FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)
X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

# 1. Start a nameserver
NS = hpns.NameServer(run_id='example2', host='127.0.0.1', port=None)
NS.start()

# 2. Start the workers
workers = []
for i in range(numWorkers):
    w = RandomForestWorker(X_train, X_val, y_train, y_val, sleep_interval=0.5, nameserver='127.0.0.1',
                           run_id='example2', id=i)
    w.run(background=True)
    workers.append(w)

# 3. Run an optimizer
bohb = BOHB(configspace=w.get_configspace(), run_id='example2', min_budget=1, max_budget=9, eta=3)
res = bohb.run(n_iterations=10, min_n_workers=numWorkers)

# 4. Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# 5. Analysis
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
