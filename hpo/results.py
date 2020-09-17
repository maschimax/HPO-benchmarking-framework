class TuningResult():
    def __init__(self, ids=None, timestamps=None, values=None, configurations=None, best_value=None,
                 best_configuration=None):
        self.ids = ids
        self.timestamps = timestamps
        self.values = values
        self.configurations = configurations
        self.best_value = best_value
        self.best_configuration = best_configuration
