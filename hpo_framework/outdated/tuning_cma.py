import cma

from datasets.dummy import preprocessing as pp

FOLDER = r'C:\Users\Max\Documents\GitHub\housing_regression\datasets'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv'

train_raw = pp.load_data(FOLDER, TRAIN_FILE)
test_raw = pp.load_data(FOLDER, TEST_FILE)

X_train, y_train, X_val, y_val, X_test = pp.process(train_raw, test_raw, standardization=False, logarithmic=False,
                                                    count_encoding=False)


def objective_rf(X_train, y_train, X_val, y_val, params):
    # Add function body here
    # rf_reg = RandomForestRegressor(**params, random_state=0)
    val_loss = 0
    return val_loss


# Initialize optimizer
es = cma.CMAEvolutionStrategy()

# Start optimization
es.optimize(objective_fct=objective_rf, maxfun=100, n_jobs=-1) # Minimizes the objective function

