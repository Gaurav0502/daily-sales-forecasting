import warnings
warnings.filterwarnings("ignore")

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_optimizer import Ranger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters)

params = {
    "epochs": 50,
    "accelerator": "gpu",
    "gradient_clip_val": 0.03,
    "limit_train_batches": 50,
    "learning_rate": 0.01,
    "hidden_size": 64,
    "attention_head_size": 3,
    "dropout": 0.2,
    "optimiser": Ranger,
    "loss": MAPE(),
    "reduce_on_plateau_patience": 4,
    "log_interval": 10
}

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

    def __init__(self, params: dict[str, any]):
        
        self.config = params

        self.training = None
        self.validation = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.trainer = None

    def preprocess(self, data: pd.DataFrame) -> None:

        max_prediction_length = 5
        max_encoder_length = 24
        batch_size = 64
        training_cutoff = data["time_idx"].max() - max_prediction_length

        self.training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx = "time_idx",
            target = "TotalSales",
            group_ids = ["cluster_id"],
            min_encoder_length = max_encoder_length // 2,
            max_encoder_length = max_encoder_length,
            min_prediction_length = 1,
            max_prediction_length = max_prediction_length,
            allow_missing_timesteps = True
        )

        self.validation = TimeSeriesDataSet.from_dataset(
            dataset = self.training, 
            data = data, predict = True, 
            stop_randomization = True
        )

        self.train_dataloader = self.training.to_dataloader(
            train = True, 
            batch_size = batch_size, 
            num_workers = 0
        )

        self.val_dataloader = self.validation.to_dataloader(
            train = False, 
            batch_size = batch_size * 10, 
            num_workers = 0
        )

    def train(self, callbacks: list = callbacks) -> None:

        logger = TensorBoardLogger("lightning_logs")
        
        self.trainer = pl.Trainer(
            max_epochs = self.config["epochs"],
            accelerator= self.config["accelerator"],
            gradient_clip_val = self.config["gradient_clip_val"],
            limit_train_batches = self.config["limited_train_batches"],
            callbacks = callbacks,
            logger = logger,
            enable_model_summary = True
        )

        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate = self.config["learning_rate"],
            hidden_size = self.config["hidden_size"],
            attention_head_size = self.config["attention_head_size"],
            dropout = self.config["dropout"],
            hidden_continuous_size = self.config["hidden_size"]//2,
            loss = self.config["loss"],
            log_interval = self.config["log_interval"],
            optimizer = self.config["optimizer"],
            reduce_on_plateau_patience = self.config["reduce_on_plateau_patience"]
        )

        self.trainer.fit(
            self.tft,
            train_dataloaders = self.train_dataloader,
            val_dataloaders = self.val_dataloader)

    def predict(self):
        pass

    