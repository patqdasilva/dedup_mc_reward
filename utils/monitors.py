import os
import gc
import re
import psutil
import random
import subprocess
import random
import logging
from logging import Logger
from typing import Any

import torch

def make_logger(fp: str) -> Logger:
    """Make a logger object to write messages to a file.

    Args:
        fp: file path of the log file to create

    Returns:
        logger: used to write messages to the file
    """
    # create a random name to lower chances
    # of saving duplicate info if running twice without refreshing kernel
    random_name = str(random.random())
    # Create and configure logger
    logger = logging.getLogger(random_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(fp))
    logger.handlers[0].setFormatter(logging.Formatter(
        '%(asctime)s\t%(message)s', datefmt='%Y/%m/%d\t%I:%M:%S %p'
    ))
    return logger

def get_RAM_usage(message: str) -> str:
    """Get RAM usage.
    
    Args:
        message: message to write in log
    Returns:
        ram_use: formatted message for log
    """
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memory_info = python_process.memory_info()
    ram_use = (
        f'RAM: {message}\t'
        f'{round(memory_info.rss / 2**20, 2)} MB'
    )
    return ram_use

def get_VRAM_usage(message: str) -> str:
    """Get VRAM usage for a single GPU.
    
    Args:
        message: message to write in log
    
    Returns:
        vram_use: formmated message for log
    """
    
    result = subprocess.run(
        'nvidia-smi',
        shell=True, capture_output=True, text=True
    )
    pattern = r'(\d+)MiB \/ .*?(\d+)MiB'
    match = re.search(pattern, result.stdout)
    vram_use = f'VRAM: {message}\t{match.group()}'
    return vram_use

def manage_mem(*vars: Any, force_gc: bool = False) -> None:
    """Free variables from memory.

    Args:
        vars: any variables to delete
        force_gc: whether or not to collect garbage
    Returns:
        None    
    """
    for variable in vars:
        del variable
    torch.cuda.empty_cache()  # GPU
    if force_gc:
        gc.collect()  # CPU
    return None