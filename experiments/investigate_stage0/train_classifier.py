from f3dasm.datageneration import DataGenerator
from f3dasm import ExperimentData

class ClassifierConstructer(DataGenerator):
    def __init__(self, training_data: ExperimentData, testing_data: ExperimentData):
        self.training_data = training_data
        self.testing_data = testing_data

        
    def execute(self):
        """
        :)
        """
        ...
