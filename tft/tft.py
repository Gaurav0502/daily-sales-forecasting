import warnings
warnings.filterwarnings("ignore")

import copy
from pathlib import Path
import warnings

from sklearn.metrics import mean_absolute_percentage_error

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_optimizer import Ranger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAPE

import configs

lr_logger = LearningRateMonitor()
early_stop_callback = EarlyStopping(
                        monitor = "val_loss", min_delta = 1e-4, 
                        atience = 10, verbose = False, mode = "min"
                      )

callbacks = [
    early_stop_callback,
    lr_logger
]    

class TFT:

    def __init__(self, data, configs: dict[str, any]):
        
        self.data = data
        self.cv_configs = configs["cross_validation"]
        self.dataset_configs = configs["dataset"]
        self.tft_configs = configs["tft"]

        self.training = None
        self.trainer = None
        self.tft = None
        self.cv_results = None

    def __get_dataloaders(self, train_data, val_data, test_data = False):

        self.training = TimeSeriesDataSet(
            data = train_data,
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
            data = val_data, 
            predict = True, 
            stop_randomization = True
        ) 

        test = TimeSeriesDataSet.from_dataset(
            dataset = self.training, 
            data = test_data,
            predict = True, 
            stop_randomization = True
        )

        train_loader = self.training.to_dataloader(train = True, 
                                              batch_size = self.dataset_configs["batch_size"], 
                                              num_workers=0)
        
        val_loader = validation.to_dataloader(train = False, 
                                              batch_size = self.dataset_configs["batch_size"] * 10, 
                                              num_workers = 0)
        
        test_loader = test.to_dataloader(train = False, 
                                         batch_size = self.dataset_configs["batch_size"] * 10, 
                                         num_workers = 0)
        
        return train_loader, val_loader, test_loader
    
    def train(self, dataloaders: list, additional_callbacks: list = []) -> None:

        early_stop_callback = EarlyStopping(
            monitor = self.tft_configs.early_stopping_monitor, 
            min_delta = self.tft_configs.early_stopping_min_delta, 
            patience = self.tft_configs.early_stopping_patience,
            mode = self.tft_configs.early_stopping_mode,
            verbose=False
        )

        lr_logger = LearningRateMonitor()

        callbacks = [lr_logger,
                     early_stop_callback] + additional_callbacks
        
        logger = TensorBoardLogger("lightning_logs")

        self.trainer = pl.Trainer(
            max_epochs = self.tft_configs.max_epochs,
            accelerator = self.tft_configs.accelarator,
            gradient_clip_val = self.tft_configs.gradient_clip_val,
            limit_train_batches = self.tft_configs.limit_train_batches,
            callbacks = callbacks,
            logger = logger,
            enable_model_summary = False,
            enable_progress_bar=False
        )

        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate = self.tft_configs.lr,
            hidden_size = self.tft_configs.hidden_size,
            attention_head_size = self.tft_configs.attention_head_size,
            dropout = self.tft_configs.dropout,
            hidden_continuous_size = self.tft_configs.hidden_continuous_size,
            loss = self.tft_configs.loss,
            log_interval = 10,
            optimizer = self.tft_configs.optimizer,
            reduce_on_plateau_patience = self.tft_configs.reduce_on_plateau_patience,
        )

        self.trainer.fit(
            model = self.tft,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"]
        )

    def cross_validate(self, MIN_TIME_IDX: int, MAX_TIME_IDX: int):

        total_window_width = self.cv_configs.train + self.cv_configs.val + self.cv_configs.test
        self.cv_results = []

        for start_idx in range(MIN_TIME_IDX, MAX_TIME_IDX - total_window_width + 1, self.cv_configs.stride):

            train_end = start_idx + self.cv_configs.train
            val_end = train_end + self.cv_configs.val
            test_end = val_end + self.cv_configs.test

            train_data = self.data[(self.data["time_idx"] >= start_idx) & (self.data["time_idx"] < train_end)]

            val_data = self.data[
                (self.data["time_idx"] >= train_end - self.dataset_configs.max_encoder_length) & (self.data["time_idx"] < val_end)
            ]

            test_data = self.data[
                (self.data["time_idx"] >= val_end - self.dataset_configs.max_encoder_length) & (self.data["time_idx"] < test_end)
            ]

            train_dataloader, val_dataloader, test_dataloader = self.__get_dataloaders(
                                                                        train_data = train_data,
                                                                        val_data = val_data,
                                                                        test_data = test_data
                                                                    )
            
            self.train(
                dataloaders = {
                    "train": train_dataloader,
                    "val": val_dataloader
                }
            )

            actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
            predictions = self.tft.predict(test_dataloader, 
                                           trainer_kwargs = dict(accelerator = self.tft_configs.accelarator))
            
            self.cv_results.append({
                "train_range": (start_idx, train_end),
                "val_range": (train_end, val_end),
                "test_range": (val_end, test_end),
                "actuals": actuals,
                "predictions": predictions,
                "MAPE": mean_absolute_percentage_error(actuals.cpu(), 
                                                       predictions.cpu()) * 100
            })


    def predict(self, dataloader: any):
        
        return self.tft.predict(dataloader)
    