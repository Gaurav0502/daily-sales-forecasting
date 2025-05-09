# for data only classes
from dataclasses import dataclass


@dataclass
class CrossValConfigs:
    train: int = 120
    val: int = 5
    test: int = 5
    stride: int = 5


@dataclass
class DataSetConfigs:
    time_idx: str = "time_idx"
    target: str = "TotalSales"
    group_ids: list = ("cluster_id",)
    encoder_length: int = 12
    prediction_length: int = 5
    batch_size: int = 64  

    add_relative_time_idx = False
    add_target_scales = True
    add_encoder_length = "auto"

@dataclass
class OptimizerConfigs:
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 10
    early_stopping_mode: str = "min"
    early_stopping_min_delta: float = 1e-4

    max_epochs: int = 50
    accelerator: str = "gpu"
    gradient_clip_val: float = 0.1
    limit_train_batches: int = 50

@dataclass
class NHiTSConfigs:
    lr: float = 0.001
    hidden_size: int = 256
    dropout: float = 0.2
    shared_weights: bool = True
    n_blocks = [2, 2, 2]
    n_layers: int = 2
    activation: str = "ReLU"
    
    reduce_on_plateau_patience: int = 4