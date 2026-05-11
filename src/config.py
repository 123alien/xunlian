"""Central configuration — dataclass + YAML loading."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    window_size: int = 24
    horizon: int = 1
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class ModelConfig:
    type: str = "lstm"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    pretrain_epochs: int = 50
    finetune_epochs: int = 30
    finetune_lr: float = 1e-4


@dataclass
class AnomalyConfig:
    mad_window: int = 24
    threshold: float = 3.0
    epsilon: float = 1e-8


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            anomaly=AnomalyConfig(**d.get("anomaly", {})),
            seed=d.get("seed", 42),
        )

    def to_dict(self) -> dict:
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "train": self.train.__dict__,
            "anomaly": self.anomaly.__dict__,
            "seed": self.seed,
        }
