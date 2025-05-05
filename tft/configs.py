import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from pytorch_optimizer import Ranger
from pytorch_forecasting.metrics import MAPE, Metric


@dataclass
class CrossValConfigs:
    train: int = 120
    val: int = 5
    test: int = 5
    stride: int = val + test


@dataclass
class DataSetConfigs:
    time_idx: str = "time_idx"
    target: str = "TotalSales"
    group_ids: list = ("cluster_id",)
    max_encoder_length: int = 24
    min_prediction_length: int = 1
    max_prediction_length: int = 5
    batch_size: int = 64


@dataclass
class TFTConfigs:
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 10
    early_stopping_mode: str = "min"
    early_stopping_min_delta: float = 1e-4

    max_epochs: int = 50
    accelerator: str = "gpu"
    gradient_clip_val: float = 0.1
    limit_train_batches: int = 50

    lr: float = 0.001
    hidden_size: int = 32
    attention_head_size: int = 2
    dropout: float = 0.2
    loss: Metric = MAPE()
    optimizer = Ranger
    reduce_on_plateau_patience: int = 4




