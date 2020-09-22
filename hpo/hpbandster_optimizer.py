import time
import hpbandster.core.nameserver as hpns

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult


class HpbandsterOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, budget,
                         random_seed)

    def optimize(self) -> TuningResult:
        """

        :return:
        """
        # Step 1: Start a nameserver
        NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
        NS.start()

        # Step 2: Start a worker
        worker = RandomForestWorker(X_train, X_val, y_train, y_val,
                                nameserver='127.0.0.1', run_id='example1')
        worker.run(background=True)

        # Step 3: Run an optimizer
        optimizer = BOHB(configspace=worker.get_configspace(), run_id='example1',
                         nameserver='127.0.0.1', min_budget=1, max_budget=9, eta=3.0,
                         result_logger=result_logger)
        res = optimizer.run(n_iterations=10)


        optimizer.shutdown(shutdown_workers=True)
        NS.shutdown()

        pass

    def objective(self):
        pass
