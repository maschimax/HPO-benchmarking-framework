import time
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.optimizers import HyperBand
import pandas as pd

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult
from hpo.hpbandster_worker import HPBandsterWorker


class HpbandsterOptimizer(BaseOptimizer):
    def __init__(self, hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                 random_seed):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed)

    def optimize(self) -> TuningResult:
        """

        :return:
        """
        # Step 1: Start a nameserver
        NS = hpns.NameServer(run_id='hpbandster', host='127.0.0.1', port=None)
        NS.start()

        # Step 2: Start a worker
        worker = HPBandsterWorker(x_train=self.x_train, x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                  ml_algorithm=self.ml_algorithm, optimizer_object=self,
                                  nameserver='127.0.0.1', run_id='hpbandster')

        worker.run(background=True)

        # Step 3: Run an optimizer
        # Select the specified HPO-tuning method
        if self.hpo_method == 'BOHB':
            eta = 3.0
            optimizer = BOHB(configspace=worker.get_configspace(self.hp_space), run_id='hpbandster',
                             nameserver='127.0.0.1', min_budget=1, max_budget=10, eta=eta)
            # Values for budget stages: https://arxiv.org/abs/1905.04970

        elif self.hpo_method == 'Hyperband':
            eta = 3.0
            optimizer = HyperBand(configspace=worker.get_configspace(self.hp_space), run_id='hpbandster',
                                  nameserver='127.0.0.1', min_budget=1, max_budget=10, eta=eta)
            # Values for budget stages: https://arxiv.org/abs/1905.04970

        else:
            raise Exception('Unknown HPO-method!')

        # Optimize on the predefined n_fßunc_evals and measure the wall clock times
        start_time = time.time()
        # >>> NECESSARY FOR HPBANDSTER?
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        res = optimizer.run(n_iterations=int(self.n_func_evals / eta))
        # Relation of budget stages, halving iterations and the number of evaluations: https://arxiv.org/abs/1905.04970
        # number of function evaluations = eta * n_iterations

        # >>> USE HPBANDSTER'S CAPABILITIES FOR TIME MEASUREMENT INSTEAD?
        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        # wall_clock_time = max(self.times)

        optimizer.shutdown(shutdown_workers=True)
        NS.shutdown()

        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()

        best_params = id2config[incumbent]['config']

        runs_df = pd.DataFrame(columns=['config_id#0', 'config_id#1', 'config_id#2', 'iteration', 'budget',
                                        'loss', 'timestamps [finished]'])
        all_runs = res.get_all_runs()

        for i in range(len(all_runs)):
            this_run = all_runs[i]
            temp_dict = {'run_id': [i],
                         'config_id#0': [this_run.config_id[0]],
                         'config_id#1': [this_run.config_id[1]],
                         'config_id#2': [this_run.config_id[2]],
                         'iteration': this_run.config_id[0],
                         'budget': this_run.budget,
                         'loss': this_run.loss,
                         'timestamps [finished]': self.times[i]}
            # alternatively: 'timestamps [finished]': this_run.time_stamps['finished']
            this_df = pd.DataFrame.from_dict(data=temp_dict)
            this_df.set_index('run_id', inplace=True)
            runs_df = pd.concat(objs=[runs_df, this_df], axis=0)

        runs_df.sort_values(by=['timestamps [finished]'], ascending=True, inplace=True)

        losses = list(runs_df['loss'])
        best_loss = min(losses)
        evaluation_ids = list(range(1, len(losses) + 1))
        timestamps = list(runs_df['timestamps [finished]'])
        wall_clock_time = max(timestamps)

        configurations = ()
        for i in range(len(losses)):
            this_config = (list(runs_df['config_id#0'])[i],
                           list(runs_df['config_id#1'])[i],
                           list(runs_df['config_id#2'])[i])

            configurations = configurations + (id2config[this_config]['config'],)

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_params,
                              wall_clock_time=wall_clock_time)

        return result
