from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import skopt


class HPBandsterWorker(Worker):
    def __init__(self, ml_algorithm, optimizer_object, sleep_interval=0, **kwargs):
        super().__init__(**kwargs)

        self.ml_algorithm = ml_algorithm
        self.optimizer_object = optimizer_object
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, *args, **kwargs):  # ersetzt die objective function aus der optimizer Klasse

        try:
            # Pass the Hyperband budget (hpbandster specific) to the evaluation function
            val_loss = self.optimizer_object.train_evaluate_ml_model(params=config, hb_budget=budget,
                                                                     cv_mode=self.optimizer_object.cross_val,
                                                                     test_mode=False)
            training_successful = True

        # If the training fails (algorithm crash)
        except:
            val_loss = float('nan')
            training_successful = False

        return ({'loss': val_loss,
                 'info': training_successful})

    @staticmethod
    def get_configspace(hp_space: dict):
        cs = CS.ConfigurationSpace()

        # Transform the skopt hyperparameter space into the required format for hpbandster
        params_list = []
        for i in range(len(hp_space)):

            if type(hp_space[i]) == skopt.space.space.Integer:
                params_list.append(CSH.UniformIntegerHyperparameter(name=hp_space[i].name,
                                                                    lower=hp_space[i].low,
                                                                    upper=hp_space[i].high))

            elif type(hp_space[i]) == skopt.space.space.Categorical:
                params_list.append(CSH.CategoricalHyperparameter(hp_space[i].name,
                                                                 choices=list(hp_space[i].categories)))

            elif type(hp_space[i]) == skopt.space.space.Real:
                # Sample in the log domain
                if hp_space[i].prior == 'log-uniform' and hp_space[i].base == 10:
                    params_list.append(CSH.UniformFloatHyperparameter(hp_space[i].name,
                                                                      lower=hp_space[i].low,
                                                                      upper=hp_space[i].high,
                                                                      log=True))
                # Uniform sampling
                else:
                    params_list.append(CSH.UniformFloatHyperparameter(hp_space[i].name,
                                                                      lower=hp_space[i].low,
                                                                      upper=hp_space[i].high,
                                                                      log=False))

            else:
                raise Exception('The skopt HP-space could not be converted correctly!')

        cs.add_hyperparameters(params_list)

        return cs

    @staticmethod
    def get_warmstart_config(hp_space: dict, warmstart_params: dict):

        ws_cs = CS.ConfigurationSpace()

        ws_params_list = []
        # Create a hpbandster hyperparameter space with the warmstart hyperparameters
        for i in range(len(hp_space)):
            this_param = hp_space[i].name

            if type(hp_space[i]) == skopt.space.space.Integer:
                # ConfigSpace doesn't accept equal values for the lower and the upper bounds (integer HPs)
                if warmstart_params[this_param] == hp_space[i].high:
                    ws_params_list.append(CSH.UniformIntegerHyperparameter(name=this_param,
                                                                           lower=warmstart_params[this_param] - 1,
                                                                           upper=warmstart_params[this_param]))

                else:
                    ws_params_list.append(CSH.UniformIntegerHyperparameter(name=this_param,
                                                                           lower=warmstart_params[this_param],
                                                                           upper=warmstart_params[this_param] + 1))

            elif type(hp_space[i]) == skopt.space.space.Categorical:
                ws_params_list.append(CSH.CategoricalHyperparameter(this_param,
                                                                    choices=[warmstart_params[this_param]]))

            elif type(hp_space[i]) == skopt.space.space.Real:
                # ConfigSpace doesn't accept equal values for the lower and the upper bounds (real HPs)
                if warmstart_params[this_param] == hp_space[i].high:
                    ws_params_list.append(CSH.UniformFloatHyperparameter(this_param,
                                                                         lower=warmstart_params[this_param] - 0.0001,
                                                                         upper=warmstart_params[this_param]))
                else:
                    ws_params_list.append(CSH.UniformFloatHyperparameter(this_param,
                                                                         lower=warmstart_params[this_param],
                                                                         upper=warmstart_params[this_param] + 0.0001))

            else:
                raise Exception("The warmstart configuration space couldn't be created correctly.")

        ws_cs.add_hyperparameters(ws_params_list)

        return ws_cs
