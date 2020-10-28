import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

cs = CS.ConfigurationSpace()

booster = CSH.CategoricalHyperparameter('booster', ['gbtree', 'gblinear', 'dart'])
cs.add_hyperparameter(booster)

eta = CSH.UniformFloatHyperparameter('eta', lower=0, upper=1, default_value=0.3, log=False)
cs.add_hyperparameter(eta)

cond1 = CS.InCondition(eta, booster, ['gbtree', 'dart'])
cs.add_condition(cond1)

bla = 0

print(cs._hyperparameters.keys())





