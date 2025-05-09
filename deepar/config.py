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
    max_encoder_length: int = 24
    min_prediction_length: int = 1
    max_prediction_length: int = 5
    batch_size: int = 64


@dataclass
class DeepARConfigs:
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 20
    early_stopping_mode: str = "min"
    early_stopping_min_delta: float = 1e-4

    max_epochs: int = 100
    accelerator: str = "gpu"
    gradient_clip_val: float = 0.1
    limit_train_batches: int = 1.0

    lr: float = 0.001
    cell_type: str = "lstm"
    hidden_size: int = 30
    rnn_layers: int = 3
    dropout: float = 0.2
    loss_rank: int = 20