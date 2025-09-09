"""
Base data stage
"""

import torch
from typing import Union, Dict
from src.stage.base import Execute

class DataStage(Execute):
    """
    Base dataset stage for vision and language datasets
    """
    def __init__(self, config_dir: Union[str, Dict]):
        if not isinstance(config_dir, (str, dict)):
            raise TypeError(f"config_dir must be either a string (path) or dict, got {type(config_dir)}")
            
        super().__init__(config_dir)

        if not isinstance(self.config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(self.config)}")
            
        if "dataset" not in self.config:
            raise KeyError("Config must contain 'dataset' key")
            
        if "name" not in self.config["dataset"]:
            raise KeyError("Config must contain 'dataset.name' key")
            
        if "split" not in self.config["dataset"]:
            raise KeyError("Config must contain 'dataset.split' key")
            
        if "train" not in self.config:
            raise KeyError("Config must contain 'train' key")
            
        if "batch_size" not in self.config["train"]:
            raise KeyError("Config must contain 'train.batch_size' key")

        self.dataset_name = self.config["dataset"]["name"]
        self.data_split = self.config["dataset"]["split"]
        self.batch_size = self.config["train"]["batch_size"]

        # ddp flag
        self.is_ddp = torch.distributed.is_initialized()

    def __len__(self):
        return len(self.dataset)

    def __name__(self):
        return "BaseDataStage"

    def load_dataset(self):
        return []

    def prepare_transform(self):
        pass

    def prepare_loader(self):
        pass

    def run(self):
        self.logger.info(f"Preparing dataset {self.dataset_name}")