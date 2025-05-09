# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# standard libraries
import copy
from pathlib import Path

# progress bar
from tqdm import tqdm

# numerical and data handling
import numpy as np
import pandas as pd

# evaluation metrics
from sklearn.metrics import mean_absolute_percentage_error

# signal smoothing
from scipy.signal import savgol_filter

# pytorch lightning utilities
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# pytorch forecasting core
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.nhits import NHiTS

# normalizers
from pytorch_forecasting.data import GroupNormalizer

# forecasting loss functions
from pytorch_forecasting.metrics import MAPE, MAE, QuantileLoss

# optimizer
from pytorch_optimizer import Ranger

# experiment logging
import wandb

# function typing
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Any


class NHiTS_model:

    def __init__(self, data: pd.DataFrame, loss: Any, 
                 optimizer: Any, configs: dict[str, Any]):
        
        self.data = data
        self.cv_configs = configs["cross_validation"]
        self.dataset_configs = configs["dataset"]
        self.optimizer_configs = configs["optimizer"]
        self.nhits_configs = configs["nhits"]

        self.training = None
        self.trainer = None
        self.nhits = None
        self.cv_results = None
        self.loss = loss
        self.optimizer = optimizer
       
    def apply_savgol_filter(self, x: pd.Series) -> Any:
        return savgol_filter(x, window_length = 9, polyorder = 2)
        
    def __get_dataloaders(self, train_data: pd.DataFrame, 
                          val_data: pd.DataFrame, 
                          test_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader, 
                                                            DataLoader]:
        
        train_data["TotalSales"] = (
            train_data
            .groupby("cluster_id")["TotalSales"]
            .transform(self.apply_savgol_filter)
        )

        val_data["TotalSales"] = (
            val_data
            .groupby("cluster_id")["TotalSales"]
            .transform(self.apply_savgol_filter)
        )

        test_data["TotalSales"] = (
            test_data
            .groupby("cluster_id")["TotalSales"]
            .transform(self.apply_savgol_filter)
        )
        
        self.training = TimeSeriesDataSet(
            data = train_data,
            time_idx = self.dataset_configs.time_idx,
            target = self.dataset_configs.target,
            group_ids = self.dataset_configs.group_ids,
            min_encoder_length = self.dataset_configs.encoder_length,
            max_encoder_length = self.dataset_configs.encoder_length,
            min_prediction_length = self.dataset_configs.prediction_length,
            max_prediction_length = self.dataset_configs.prediction_length,

            time_varying_known_reals = ["time_idx", "days_to_christmas"],
            time_varying_unknown_reals = ["TotalSales"],
            static_categoricals = ["cluster_id"],
            target_normalizer = GroupNormalizer(groups = ["cluster_id"], 
                                                method = "robust"),
            add_relative_time_idx = self.dataset_configs.add_relative_time_idx,
            add_target_scales = self.dataset_configs.add_target_scales,
            add_encoder_length = self.dataset_configs.add_encoder_length
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
                                              batch_size = self.dataset_configs.batch_size, 
                                              num_workers = 0)
        
        val_loader = validation.to_dataloader(train = False, 
                                              batch_size = self.dataset_configs.batch_size * 10, 
                                              num_workers = 0)
        
        test_loader = test.to_dataloader(train = False, 
                                         batch_size = self.dataset_configs.batch_size * 10, 
                                         num_workers = 0)
        
        return train_loader, val_loader, test_loader
        
    def train(self, dataloaders: list, log_id: int, 
              additional_callbacks: list = []) -> None:

        early_stop_callback = EarlyStopping(
            monitor = self.optimizer_configs.early_stopping_monitor, 
            min_delta = self.optimizer_configs.early_stopping_min_delta, 
            patience = self.optimizer_configs.early_stopping_patience,
            mode = self.optimizer_configs.early_stopping_mode,
            verbose=False
        )

        lr_logger = LearningRateMonitor()

        callbacks = [lr_logger,
                     early_stop_callback] + additional_callbacks
        
        logger = WandbLogger(
                    project = "NHiTS Window-based Evaluation",
                    name = f"window_{log_id}",
                    log_model = True
                )

        self.trainer = pl.Trainer(
            max_epochs = self.optimizer_configs.max_epochs,
            accelerator = self.optimizer_configs.accelerator,
            gradient_clip_val = self.optimizer_configs.gradient_clip_val,
            limit_train_batches = self.optimizer_configs.limit_train_batches,
            callbacks = callbacks,
            logger = logger,
            enable_model_summary = False,
            enable_progress_bar = False
        )

        self.nhits = NHiTS.from_dataset(
            self.training,
            learning_rate = self.nhits_configs.lr,
            hidden_size = self.nhits_configs.hidden_size,
            dropout = self.nhits_configs.dropout,
            shared_weights = self.nhits_configs.shared_weights,
            n_blocks = self.nhits_configs.n_blocks,
            n_layers = self.nhits_configs.n_layers,
            activation = self.nhits_configs.activation,
            
            loss = self.loss,
            log_interval = 10,
            optimizer = self.optimizer,
            reduce_on_plateau_patience = self.nhits_configs.reduce_on_plateau_patience,
        )


        self.trainer.fit(
            model = self.nhits,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"]
        )
        
    def cross_validate(self, MIN_TIME_IDX: int, MAX_TIME_IDX: int) -> None:

        total_window_width = self.cv_configs.val + self.cv_configs.test
        self.cv_results = []
        i = 0

        for k in tqdm(range(MIN_TIME_IDX, 
                               MAX_TIME_IDX - total_window_width + 1, 
                               self.cv_configs.stride),
                             desc = "cross-validation",
                             unit = "window"):

            wandb.init(project="NHiTS Window-based Evaluation", 
                       name = f"window_{i}",
                       settings = wandb.Settings(quiet = True))

            start_idx = k - MIN_TIME_IDX

            train_end = start_idx + self.cv_configs.train
            val_end = train_end + self.cv_configs.val
            test_end = val_end + self.cv_configs.test

            train_data = self.data[(self.data["time_idx"] >= 0) & (self.data["time_idx"] < train_end)]

            val_data = self.data[
                (self.data["time_idx"] >= train_end - self.dataset_configs.encoder_length) & (self.data["time_idx"] < val_end)
            ]

            test_data = self.data[
                (self.data["time_idx"] >= val_end - self.dataset_configs.encoder_length) & (self.data["time_idx"] < test_end)
            ]

            train_dataloader, val_dataloader, test_dataloader = self.__get_dataloaders(
                                                                        train_data = train_data,
                                                                        val_data = val_data,
                                                                        test_data = test_data
                                                                    )
            print("data...preprocessed")
            
            self.train(
                dataloaders = {
                    "train": train_dataloader,
                    "val": val_dataloader
                },
                log_id = i
            )

            print("training...done")
            i += 1

            wandb.finish()

            actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
            predictions = self.nhits.predict(test_dataloader, 
                                             trainer_kwargs = dict(accelerator = self.nhits_configs.accelerator))
            
            self.cv_results.append({
                "train_range": (start_idx, train_end),
                "val_range": (train_end, val_end),
                "test_range": (val_end, test_end),
                "actuals": actuals,
                "predictions": predictions,
                "MAPE": mean_absolute_percentage_error(actuals.cpu(), 
                                                       predictions.cpu()) * 100
            })

            print("testing...done")
            print(f"This window got a MAPE of {self.cv_results[-1]['MAPE']}")
            
    def predict(self, dataloader: Any) -> Any:
        
        return self.nhits.predict(dataloader)