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
                 random_seed, n_workers):
        super().__init__(hp_space, hpo_method, ml_algorithm, x_train, x_val, y_train, y_val, metric, n_func_evals,
                         random_seed, n_workers)

    def optimize(self) -> TuningResult:
        """
        Method performs a hyperparameter optimization run according to the selected HPO-method.
        :return: result: TuningResult
            TuningResult-object that contains the results of this optimization run.
        """

        # Start a nameserver
        NS = hpns.NameServer(run_id='hpbandster', host='127.0.0.1', port=None)
        NS.start()

        # Start a worker
        worker = HPBandsterWorker(x_train=self.x_train, x_val=self.x_val, y_train=self.y_train, y_val=self.y_val,
                                  ml_algorithm=self.ml_algorithm, optimizer_object=self,
                                  nameserver='127.0.0.1', run_id='hpbandster')

        worker.run(background=True)

        # Run an optimizer
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

        # Optimize on the predefined n_func_evals and measure the wall clock times
        start_time = time.time()
        self.times = []  # Initialize a list for saving the wall clock times

        # Start the optimization
        try:
            res = optimizer.run(n_iterations=int(self.n_func_evals / eta))
            # Relation of budget stages, halving iterations and the number of evaluations: https://arxiv.org/abs/1905.04970
            # number of function evaluations = eta * n_iterations
            run_successful = True

            # Check whether one of the evaluations failed (hpbandster continues the optimization procedure even if
            # the objective function cannot be evaluated)
            for config_key in res.data.keys():
                this_result = res.data[config_key].results

                for this_eval in this_result.keys():
                    this_success_flag = this_result[this_eval]['info']

                    # The run wasn't successful, if one of the evaluations failed
                    if not this_success_flag:
                        run_successful = False
                        break

        # Algorithm crashed
        except:
            # Add a warning here
            run_successful = False

        # Shutdown the optimizer and the server
        optimizer.shutdown(shutdown_workers=True)
        NS.shutdown()

        # If the optimization run was successful, determine the optimization results
        if run_successful:

            for i in range(len(self.times)):
                # Subtract the start time to receive the wall clock time of each function evaluation
                self.times[i] = self.times[i] - start_time
            wall_clock_time = max(self.times)

            # Timestamps
            timestamps = self.times

            # Extract the results and create an TuningResult instance to save them
            id2config = res.get_id2config_mapping()
            incumbent = res.get_incumbent_id()

            # Best hyperparameter configuration
            best_configuration = id2config[incumbent]['config']

            runs_df = pd.DataFrame(columns=['config_id#0', 'config_id#1', 'config_id#2', 'iteration', 'budget',
                                            'loss', 'timestamps [finished]'])
            all_runs = res.get_all_runs()

            # Iterate over all runs
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

            # Sort according to the timestamps
            runs_df.sort_values(by=['timestamps [finished]'], ascending=True, inplace=True)

            losses = list(runs_df['loss'])
            best_loss = min(losses)
            evaluation_ids = list(range(1, len(losses) + 1))
            # timestamps = list(runs_df['timestamps [finished]'])  # << hpbandster's capabilities for time measurement

            configurations = ()
            for i in range(len(losses)):
                this_config = (list(runs_df['config_id#0'])[i],
                               list(runs_df['config_id#1'])[i],
                               list(runs_df['config_id#2'])[i])

                configurations = configurations + (id2config[this_config]['config'],)

        # Run not successful (algorithm crashed)
        else:
            evaluation_ids, timestamps, losses, configurations, best_loss, best_configuration, wall_clock_time = \
                self.impute_results_for_crash()

        # Pass the results to a TuningResult-object
        result = TuningResult(evaluation_ids=evaluation_ids, timestamps=timestamps, losses=losses,
                              configurations=configurations, best_loss=best_loss, best_configuration=best_configuration,
                              wall_clock_time=wall_clock_time, successful=run_successful)

        return result
