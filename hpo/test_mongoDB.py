import math
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
import os
from multiprocessing import Process

# os.mkdir("./mongo_db")
# os.system("mongod --dbpath ./mongo_db --port 1234 --directoryperdb --journal")

os.system(
    "mongod --dbpath /home/max/Desktop/Projects/housing_regression/mongodb --port 1234 --fork --logpath "
    "/home/max/Desktop/Projects/housing_regression/log_mongodb --directoryperdb --journal")


def target_func1(evals):
    trials = MongoTrials('mongo://localhost:1234/mongo_db/jobs', exp_key='exp2')
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=evals)
    return best


def target_func2():
    os.system("hyperopt-mongo-worker --mongo=localhost:1234/mongo_db --poll-interval=0.1")
    return


proc = []
for i in range(4):
    if i == 0:

        p = Process(target=target_func1, args=(10,))
    else:
        p = Process(target=target_func2)

    p.start()
    proc.append(p)

for p in proc:
    p.join()

bla = 0
