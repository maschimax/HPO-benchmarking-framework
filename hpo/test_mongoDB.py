import math
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
import os
from multiprocessing import Process

# Requirement: Select correct database in mongo shell before

def target_func1(evals):
    trials = MongoTrials('mongo://localhost:27017/mongo_hpo/jobs', exp_key='exp1')
    best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=evals)
    return best

def target_func2():
    os.system("hyperopt-mongo-worker --mongo=localhost:27017/mongo_hpo --poll-interval=0.1")
    return


proc = []
for i in range(4):
    if i == 0:

        p = Process(target=target_func1, args=(20,))
    else:
        p = Process(target=target_func2)

    p.start()
    proc.append(p)

for p in proc:
    p.join()

bla = 0
