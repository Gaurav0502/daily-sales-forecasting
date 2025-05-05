# ignoring warnings
import warnings
warnings.filterwarnings("ignore")
import copy

# for MAPE computation
from sklearn.metrics import mean_absolute_percentage_error

# for TFT modelling
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

# for optimization
from pytorch_optimizer import Ranger

# for TFT model architecture
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAPE

# for hyperparameter tuning
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

class Tuner:

    def __init__(self, data, configs: dict[str, any], param_space: dict[str, any], num_samples: int):

        self.data = data
        self.dataset_configs = configs["dataset"]
        self.tft_configs = configs["tft"]

        self.param_space = param_space
        self.num_samples = num_samples

        self.training = None
        self.train_dataloader = None
        self.val_dataloader = None

        self.trainer = None
        self.tft = None

    def __get_dataloaders(self):

        training_cutoff = self.data["time_idx"].max() - self.dataset_configs.max_prediction_length

        self.training = TimeSeriesDataSet(
            data = self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx = self.dataset_configs.time_idx,
            target = self.dataset_configs.target,
            group_ids = self.dataset_configs.group_ids,
            min_encoder_length = self.dataset_configs.max_encoder_length // 2,
            max_encoder_length = self.dataset_configs.max_encoder_length,
            min_prediction_length = self.dataset_configs.min_prediction_length,
            max_prediction_length = self.dataset_configs.max_prediction_length,
            allow_missing_timesteps = True
        )

        validation = TimeSeriesDataSet.from_dataset(
            dataset = self.training, 
            data = self.data, 
            predict = True, 
            stop_randomization = True
        )

       
        self.train_dataloader = self.training.to_dataloader(
                                        train = True, 
                                        batch_size = self.dataset_configs.batch_size, 
                                        num_workers=0
                                    )
        
        self.val_dataloader = validation.to_dataloader(
                                    train = False, 
                                    batch_size = self.dataset_configs.batch_size * 10, 
                                    num_workers = 0
                                )
    
    def train(self, config: dict[str, any]):
       
        early_stop_callback = EarlyStopping(
            monitor = self.tft_configs.early_stopping_monitor, 
            min_delta = self.tft_configs.early_stopping_min_delta, 
            patience = self.tft_configs.early_stopping_patience,
            mode = self.tft_configs.early_stopping_mode,
            verbose=False
        )

        lr_logger = LearningRateMonitor()

        tune_callback = TuneReportCallback(
                            {"val_loss": "val_loss"},
                            on = "validation_end"
                        )

        callbacks = [lr_logger,
                     early_stop_callback,
                     tune_callback]

        logger = TensorBoardLogger("lightning_logs")
        
        self.trainer = pl.Trainer(
            max_epochs = self.tft_configs.max_epochs,
            accelerator = self.tft_configs.accelerator,
            gradient_clip_val = self.tft_configs.gradient_clip_val,
            limit_train_batches = self.tft_configs.limit_train_batches,
            callbacks = callbacks,
            logger = logger,
            enable_model_summary = False,
            enable_progress_bar=False
        )
        
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=config["lr"],
            hidden_size=config["hidden_size"],
            attention_head_size = self.tft_configs.attention_head_size,
            dropout = config["dropout"],
            hidden_continuous_size = config["hidden_size"]//2,
            loss = self.tft_configs.loss,
            log_interval = 10,
            optimizer = self.tft_configs.optimizer,
            reduce_on_plateau_patience = self.tft_configs.reduce_on_plateau_patience
        )
        
        self.trainer.fit(
        self.tft,
        train_dataloaders = self.train_dataloader,
        val_dataloaders = self.val_dataloader)

    def tune(self) -> str:

        self.__get_dataloaders()
        self.analysis = tune.run(
            self.train,
            config = self.param_space,
            num_samples = self.num_samples,
            resources_per_trial = {"gpu": 1}
        )

        return self.analysis.get_best_trial(metric = "val_loss", 
                                            mode = "min")