"""
Execute Stage
"""

import os
import wandb
import logging
import yaml
import torch
import platform
from typing import Union, Dict
from datetime import datetime

class Execute:
    """
    Bottom-level execution stage.

    Args:
        config_dir: Path of configuration or config dict.
    """
    def __init__(self, config_dir: Union[str, Dict]):
        if not isinstance(config_dir, (str, dict)):
            raise TypeError(f"config_dir must be either a string (path) or dict, got {type(config_dir)}")
            
        # config dir
        self.config_dir = config_dir
        self.config = self.prepare_config()
        
        # Get node and GPU info
        node_name = self._get_node_name()
        gpu_info = self._get_gpu_info()
        
        # Check if node name is already in the path
        base_run_dir = self.config["save"]["run_dir"]
        # if node_name not in str(base_run_dir):
        #     # Add timestamp to the run directory
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     node_gpu_time = f"{node_name}_{gpu_info}_{timestamp}"
        #     # Modify run directory to include node and GPU info
        #     self.run_dir = os.path.join(base_run_dir, f"{node_gpu_time}")
        #     self.config["save"]["run_dir"] = self.run_dir
        # else:
        #     self.run_dir = base_run_dir
        self.run_dir = base_run_dir
        
        self.use_accelerate = self.config["train"].get("use_accelerate", False)

        # directory
        self.register_run_dir()

        # initialize logging
        self.logger = self.initialize_logger()

        # detect device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _get_gpu_info(self):
        """Get GPU type and count."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            return f"{gpu_count}x{gpu_name.replace(' ', '_')}"
        return "cpu"

    def _get_node_name(self):
        """Get the node/host name."""
        return platform.node().replace('.', '_')

    def initialize_logger(self):
        logname = self.config["save"]["logger"]
        logpath = os.path.join(self.run_dir, logname)

        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            file_handler = logging.FileHandler(logpath, mode="w")
            console_handler = logging.StreamHandler()
            
            file_handler.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        
        if not self.use_accelerate:
            project = self.config["wandb"]["project_name"]
            name = self.config["wandb"]["experiment_name"]
            self.wandb_flag = self.config["wandb"]["flag"]

            if self.wandb_flag:
                self.wandb_logger = wandb.init(
                    project=project,
                    name=name,
                )

        return logger

    def prepare_config(self):
        if isinstance(self.config_dir, dict):
            return self.config_dir
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"Config file not found: {self.config_dir}")
        with open(self.config_dir, 'r') as f:
            config = yaml.full_load(f)
        return config
    
    def register_run_dir(self):
        if not os.path.isdir(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)

    def run(self):
        """
        Entrance of execution
        """
        self.logger.info(f"Start stage...")