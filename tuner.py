from abc import ABC, abstractmethod

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

class Tuner(ABC):

    def __init__(self, param_space: dict[str, any], num_samples: int):

        self.param_space = param_space
        self.num_samples = num_samples

        self.model = None
        self.tune_callback = TuneReportCallback(
                {"val_loss": "val_loss"},
                on="validation_end"
            )


    @abstractmethod
    def train(self, config: dict[str, any]) -> None:
        pass

    def tune(self) -> str:

        self.analysis = tune.run(
            self.train,
            config = self.param_space,
            num_samples = self.num_samples,
            resources_per_trial = {"gpu": 1}
        )

        return self.analysis.get_best_trial(metric = "val_loss", 
                                            mode = "min")


        