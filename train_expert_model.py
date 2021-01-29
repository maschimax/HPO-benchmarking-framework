import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from math import sqrt

filepath = 'C:/Users/Max/Desktop/BenchmarkingRuleBase_v09_ML.csv'
do_expansion = True

rule_base = pd.read_csv(filepath, sep=';')

rule_base.drop(['Rule ID', 'thetas'], axis=1, inplace=True)

expanded_base = pd.DataFrame(columns=rule_base.columns)

# TODO: Check dictionaries
# User's programming ability
ur_programming_ability = {
    'low': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES'],
    'medium': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
            'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN'],
    'high': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
            'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']}

# UR: Need for model transparency
ur_transparency = {
    'yes': ['Default Values', 'Random Search'],
    'no': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
    'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']}

# Availability of a well documented library
ur_well_documented = {
    'yes': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
            'Hyperband', 'BOHB'],
    'no': ['Default Values', 'Random Search', 'GPBO', 'SMAC', 'TPE', 'CMA-ES',
            'Hyperband', 'BOHB', 'FABOLAS', 'BOHAMIANN']}

hpo_techs = ['Default Values', 'Random Search', 'CMA-ES', 'GPBO', 'SMAC', 'TPE',
            'HB', 'BOHB', 'FABOLAS', 'BOHAMIANN']

if do_expansion:

    # TODO: Expand RuleBase with UR and Robustness information
    for idx, rule in rule_base.iterrows():

        for this_ability in ['low', 'medium', 'high']:

            for this_transparency in ['yes', 'no']:

                for this_doc in ['yes', 'no']:
                    
                    # Create a copy of the original rule
                    expanded_rule = rule.copy(deep=True)

                    hpos_beliefs = {
                        'Default Values': rule['Default Values'],
                        'Random Search': rule['Random Search'],
                        'CMA-ES': rule['CMA-ES'],
                        'GPBO': rule['GPBO'],
                        'SMAC': rule['SMAC'],
                        'TPE': rule['TPE'],
                        'HB': rule['HB'],
                        'BOHB': rule['BOHB'],
                        'FABOLAS': rule['FABOLAS'],
                        'BOHAMIANN': rule['BOHAMIANN']}

                    # # 1 Identify HPOs with belief > 0
                    # rec_hpos = [hpo for hpo in hpos_beliefs.keys() if hpos_beliefs[hpo] > 0.0]

                    # 2 Identify HPOs that meet the UR requirements
                    # TODO: Check funtionality
                    ur_hpos = [hpo for hpo in hpo_techs if hpo in ur_programming_ability[this_ability]
                    if hpo in ur_transparency[this_transparency]
                    if hpo in ur_well_documented[this_doc]]
                    
                    ur_penalty = 0.1

                    for this_hpo in hpos_beliefs.keys():

                        if this_hpo in ur_hpos:

                            expanded_rule[this_hpo] = hpos_beliefs[this_hpo] + ur_penalty
                        
                        else:

                            if hpos_beliefs[this_hpo] - ur_penalty >= 0.0:
                                
                                expanded_rule[this_hpo] = hpos_beliefs[this_hpo] - ur_penalty

                            else:
                                
                                expanded_rule[this_hpo] = 0.0

                    # # Identify the common HPO techniques of both lists
                    # common_hpos = set(ur_hpos).intersection(set(rec_hpos))

                    # # Check whether there are common HPO techniques
                    # if len(common_hpos) > 0:
                    #     # Set beliefs to 1/len(common_hpos) for all common HPO techniques
                    #     for this_hpo in common_hpos:
                    #         expanded_rule[this_hpo] = 1.0 / len(common_hpos)

                    #     # Set beliefs to zero for all remaining HPO techniques
                    #     non_common_hpos = set(hpo_techs) - set(common_hpos)
                    #     for this_hpo in non_common_hpos:
                    #         expanded_rule[this_hpo] = 0.0

                    expanded_rule["User's programming ability"] = this_ability
                    expanded_rule['UR: need for model transparency'] = this_transparency
                    expanded_rule['UR: Availability of a well documented library'] = this_doc

                    # Append the rule to new rule to the expanded rule base
                    expanded_base = expanded_base.append(expanded_rule, ignore_index=True)

    # Drop high AP cols
    exp_base_woAP = expanded_base.loc[(expanded_base['UR: Anytime Performance'] != 'high'), :].copy(deep=True)

    exp_base_woAP.to_csv('C:/Users/Max/Desktop/expanded_rule_base_woAP.csv')
    expanded_base.to_csv('C:/Users/Max/Desktop/expanded_rule_base.csv')

expanded_base = pd.read_csv('C:/Users/Max/Desktop/expanded_rule_base.csv', index_col=0)

X_raw = expanded_base.copy(deep=True)
y_raw = X_raw.loc[:, hpo_techs].copy(deep=True)
X_raw.drop(hpo_techs, axis=1, inplace=True)

# >>>>
# TODO: FIX ERROR IN RULE BASE AND DELETE
y_raw.loc[:, 'Default Values'].fillna(value=0.0, inplace=True)
# >>>>

# Drop empty cols
drop_cols = ['UR: Robustness', 'Obtainability of gradients', 'Obtainability of good approximate', 'Supports parallel evaluations']
X_raw.drop(drop_cols, axis=1, inplace=True)

# Handle missing 'UR: Anytime Performance' values
X_raw.loc[:, 'UR: Anytime Performance'].fillna(value='low', inplace=True)

X_raw.loc[:, 'Special properties of loss function'].fillna(value='None', inplace=True)

# One-Hot-Encoding
X_raw = pd.get_dummies(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, shuffle=True, random_state=0)

# Impute missing values of 'Number of maximum function evaluation ...'
median_evals = X_train['Number of maximum function evaluations/ trials budget'].median()
X_train.loc[:, 'Number of maximum function evaluations/ trials budget'].fillna(value=median_evals, inplace=True)
X_test.loc[:, 'Number of maximum function evaluations/ trials budget'].fillna(value=median_evals, inplace=True)

# Standard Scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Modeling
model = RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))

print('RMSE: ', rmse)

# TODO: TEST FOR EXPERIMENTS
