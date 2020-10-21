import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
import matplotlib.pyplot as plt
import argparse

from hpo.outdated.workers import RandomForestWorker
from datasets.dummy import preprocessing as pp

parser = argparse.ArgumentParser(description='Example 3 - Local and Parallel Execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

args=parser.parse_args()


# loading data and preprocessing
FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)
X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)

if args.worker:
    w = RandomForestWorker(X_train, X_val, y_train, y_val, sleep_interval=0.5, nameserver='127.0.0.1',
                           run_id='example2')
    w.run(background=True)
    exit(0)

# 1. Start a nameserver
NS = hpns.NameServer(run_id='example2', host='127.0.0.1', port=None)
NS.start()

# 3. Run an optimizer
bohb = BOHB(configspace=RandomForestWorker.get_configspace(), run_id='example2',
            min_budget=args.min_budget, max_budget=args.max_budget,
            eta=3)

res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

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
