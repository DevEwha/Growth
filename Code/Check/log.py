# logging.py
import logging
from logging.handlers import RotatingFileHandler
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(
    name: str = "runner",
    log_dir: str = "./logs",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  

    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")

    fmt = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"로그 파일 시작: {log_file}")
    return logger
