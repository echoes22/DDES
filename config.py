from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from argparse import Namespace
import json
import yaml
import os
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True
    DATASET: str = 'kde'
    IMAGE_PATH: str = ''
    CSV_PATH: str = ''
    AUGMENTATION: bool = False
    USE_MINI_DS: bool = False
    LOAD_DS: bool = True
    PATHS: Dict[str, str] = field(default_factory=dict)

@dataclass
class TrainLRConfig:
    NAME: str = 'cosine'
    DECAY_EPOCHS: int = 5

@dataclass
class TrainConfig:
    FROM_SCRATCH: bool = False
    AUTO_RESUME: bool = False
    EVAL_MODE: bool = False
    START_EPOCH: int = 0
    EPOCHS: int = 50
    WARMUP_EPOCHS: int = 5
    BASE_LR: float = 5e-4
    WARMUP_LR: float = 5e-7
    MIN_LR: float = 5e-6
    SAMPLES_PER_EPOCH: int = 100000
    USE_AMP: bool = False
    LR_SCHEDULER: TrainLRConfig = field(default_factory=TrainLRConfig)

@dataclass
class ModelConfig:
    TYPE: str = 'convnext_unet'
    RESUME: str = ''

@dataclass
class Config:
    DATA: DataConfig = field(default_factory=DataConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    SAVE_FREQ: int = 1
    PRINT_FREQ: int = 5
    OUTPUT: str = ''
    TENSORBOARD_PATH: str = ''
    LOCAL_RANK: int = 0
    SEED: int = 42


def update_config(config: Config, args: Namespace) -> Config:
    if getattr(args, 'batch_size', None) is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, 'image_path', None) is not None:
        config.DATA.IMAGE_PATH = args.image_path
    if getattr(args, 'csv_path', None) is not None:
        config.DATA.CSV_PATH = args.csv_path
    if getattr(args, 'dataset', None) is not None:
        config.DATA.DATASET = args.dataset
    if getattr(args, 'model_type', None) is not None:
        config.MODEL.TYPE = args.model_type
    if getattr(args, 'resume_path', None) is not None:
        config.MODEL.RESUME = args.resume_path
    if getattr(args, 'output', None) is not None:
        config.OUTPUT = args.output
    if getattr(args, 'tensorboard_path', None) is not None:
        config.TENSORBOARD_PATH = args.tensorboard_path
    if getattr(args, 'max_epochs', None) is not None:
        config.TRAIN.EPOCHS = args.max_epochs

    if getattr(args, 'augmentation', None) is not None:
        config.DATA.AUGMENTATION = args.augmentation
    if getattr(args, 'use_mini_ds', None) is not None:
        config.DATA.USE_MINI_DS = args.use_mini_ds
    if getattr(args, 'load_ds', None) is not None:
        config.DATA.LOAD_DS = args.load_ds

    return config


def get_config(args: Namespace) -> Config:
    config = Config()
    return update_config(config, args)


def get_config_no_args() -> Config:
    return Config()


def load_config_from_file(config_file: str) -> Config:
    """
    Load config from YAML or JSON file.
    
    Args:
        config_file: Path to config file (.yaml or .json)
    
    Returns:
        Config object
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        elif config_file.endswith('.json'):
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file}")
    
    # Create config from dict
    config = Config()
    
    if 'data' in config_dict or 'DATA' in config_dict:
        data_cfg = config_dict.get('data') or config_dict.get('DATA', {})
        for key, value in data_cfg.items():
            if key.lower() == 'paths' or key.upper() == 'PATHS':
                if isinstance(value, dict):
                    config.DATA.PATHS = value
                continue
            if hasattr(config.DATA, key.upper()):
                setattr(config.DATA, key.upper(), value)
    
    if 'train' in config_dict or 'TRAIN' in config_dict:
        train_cfg = config_dict.get('train') or config_dict.get('TRAIN', {})
        for key, value in train_cfg.items():
            if key.lower() == 'lr_scheduler' or key.upper() == 'LR_SCHEDULER':
                # Handle nested LR_SCHEDULER config
                if isinstance(value, dict):
                    for lr_key, lr_value in value.items():
                        if hasattr(config.TRAIN.LR_SCHEDULER, lr_key.upper()):
                            setattr(config.TRAIN.LR_SCHEDULER, lr_key.upper(), lr_value)
            elif hasattr(config.TRAIN, key.upper()):
                setattr(config.TRAIN, key.upper(), value)
    
    if 'model' in config_dict or 'MODEL' in config_dict:
        model_cfg = config_dict.get('model') or config_dict.get('MODEL', {})
        for key, value in model_cfg.items():
            if hasattr(config.MODEL, key.upper()):
                setattr(config.MODEL, key.upper(), value)
    
    # Top-level config
    for key in ['save_freq', 'print_freq', 'output', 'tensorboard_path', 'seed', 'local_rank']:
        if key in config_dict:
            setattr(config, key.upper() if key.isupper() else key, config_dict[key])
        elif key.upper() in config_dict:
            setattr(config, key.upper(), config_dict[key.upper()])
    
    return config


def merge_config_with_args(config: Config, args: Namespace) -> Config:
    """
    Merge config file with command-line arguments.
    CLI arguments take precedence over config file values.
    
    Args:
        config: Config loaded from file
        args: Command-line arguments
    
    Returns:
        Merged Config object
    """
    # Only update config if argument was explicitly provided (not None)
    if getattr(args, 'batch_size', None) is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, 'image_path', None) is not None:
        config.DATA.IMAGE_PATH = args.image_path
    if getattr(args, 'csv_path', None) is not None:
        config.DATA.CSV_PATH = args.csv_path
    if getattr(args, 'dataset', None) is not None:
        config.DATA.DATASET = args.dataset
    if getattr(args, 'model', None) is not None:
        config.MODEL.TYPE = args.model
    if getattr(args, 'checkpoint', None) is not None:
        config.MODEL.RESUME = args.checkpoint
    if getattr(args, 'epochs', None) is not None:
        config.TRAIN.EPOCHS = args.epochs
    if getattr(args, 'lr', None) is not None:
        config.TRAIN.BASE_LR = args.lr
    if getattr(args, 'warmup_epochs', None) is not None:
        config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    if getattr(args, 'data_root', None) is not None:
        config.DATA.IMAGE_PATH = args.data_root
    if getattr(args, 'checkpoint_dir', None) is not None:
        config.OUTPUT = args.checkpoint_dir
    if getattr(args, 'logdir', None) is not None:
        config.TENSORBOARD_PATH = args.logdir
    if getattr(args, 'seed', None) is not None:
        config.SEED = args.seed
    if getattr(args, 'num_workers', None) is not None:
        config.DATA.NUM_WORKERS = args.num_workers
    if getattr(args, 'use_amp', None) is not None:
        config.TRAIN.USE_AMP = getattr(args, 'use_amp', False)
    if getattr(args, 'scheduler', None) is not None:
        config.TRAIN.LR_SCHEDULER.NAME = args.scheduler
    
    return config


def get_config_with_file(config_file: Optional[str] = None, args: Optional[Namespace] = None) -> Config:
    """
    Load config from file and optionally merge with CLI arguments.
    
    Args:
        config_file: Path to config file (YAML or JSON)
        args: Command-line arguments (take precedence over file)
    
    Returns:
        Merged Config object
    """
    if config_file and os.path.exists(config_file):
        config = load_config_from_file(config_file)
    else:
        config = Config()
    
    if args:
        config = merge_config_with_args(config, args)
    
    return config


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert Config dataclass to dictionary."""
    return {
        'data': asdict(config.DATA),
        'train': {
            **{k: v for k, v in asdict(config.TRAIN).items() if k != 'LR_SCHEDULER'},
            'lr_scheduler': asdict(config.TRAIN.LR_SCHEDULER)
        },
        'model': asdict(config.MODEL),
        'save_freq': config.SAVE_FREQ,
        'print_freq': config.PRINT_FREQ,
        'output': config.OUTPUT,
        'tensorboard_path': config.TENSORBOARD_PATH,
        'local_rank': config.LOCAL_RANK,
        'seed': config.SEED,
    }


def save_config_to_file(config: Config, output_file: str) -> None:
    """
    Save config to YAML or JSON file.
    
    Args:
        config: Config object to save
        output_file: Path to output file (.yaml or .json)
    """
    config_dict = config_to_dict(config)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        if output_file.endswith('.yaml') or output_file.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif output_file.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file format: {output_file}")
