import time
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
import pandas as pd

from hpo.baseoptimizer import BaseOptimizer
from hpo.results import TuningResult
from hpo.hpbandster_worker import HPBandsterWorker


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
        worker = HPBandsterWorker(x_train=self.x_train, x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                  ml_algorithm=self.ml_algorithm, optimizer_object=self,
                                  nameserver='127.0.0.1', run_id='example1')

        worker.run(background=True)

        # Step 3: Run an optimizer
        # >>> BUDGET <<<
        optimizer = BOHB(configspace=worker.get_configspace(self.hp_space), run_id='example1',
                         nameserver='127.0.0.1', min_budget=1, max_budget=9, eta=3.0)

        # Optimize on the predefined budget and measure the wall clock times
        start_time = time.time()
        # >>> NECESSARY FOR HPBANDSTER?
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        # >>> n_iterations ??? <<<<
        res = optimizer.run(n_iterations=int(self.budget / 3))  # set the number of iterations on the trial level?

        # >>> USE HPBANDSTER'S CAPABILITIES FOR TIME MEASUREMENT INSTEAD?
        for i in range(len(self.times)):
            # Subtract the start time to receive the wall clock time of each function evaluation
            self.times[i] = self.times[i] - start_time
        wall_clock_time = max(self.times)

        optimizer.shutdown(shutdown_workers=True)
        NS.shutdown()

        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        # inc_trajectory = res.get_incumbent_trajectory(
        #     bigger_is_better=False)  # Reconsider the use of the 'bigger-is-better' Flag

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
                         'timestamps [finished]': this_run.time_stamps['finished']}
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
